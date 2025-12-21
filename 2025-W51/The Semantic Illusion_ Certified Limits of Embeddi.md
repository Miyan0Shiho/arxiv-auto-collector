# The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems

**Authors**: Debu Sinha

**Published**: 2025-12-17 04:22:28

**PDF URL**: [https://arxiv.org/pdf/2512.15068v1](https://arxiv.org/pdf/2512.15068v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems remain susceptible to hallucinations despite grounding in retrieved evidence. Current detection methods rely on semantic similarity and natural language inference (NLI), but their fundamental limitations have not been rigorously characterized. We apply conformal prediction to hallucination detection, providing finite-sample coverage guarantees that enable precise quantification of detection capabilities. Using calibration sets of approximately 600 examples, we achieve 94% coverage with 0% false positive rate on synthetic hallucinations (Natural Questions). However, on three real hallucination benchmarks spanning multiple LLMs (GPT-4, ChatGPT, GPT-3, Llama-2, Mistral), embedding-based methods - including state-of-the-art OpenAI text-embedding-3-large and cross-encoder models - exhibit unacceptable false positive rates: 100% on HaluEval, 88% on RAGTruth, and 50% on WikiBio. Crucially, GPT-4 as an LLM judge achieves only 7% FPR (95% CI: [3.4%, 13.7%]) on the same data, proving the task is solvable through reasoning. We term this the "semantic illusion": semantically plausible hallucinations preserve similarity to source documents while introducing factual errors invisible to embeddings. This limitation persists across embedding architectures, LLM generators, and task types, suggesting embedding-based detection is insufficient for production RAG deployment.

## Full Text


<!-- PDF content starts -->

The Semantic Illusion: Certified Limits of
Embedding-Based Hallucination Detection in RAG
Systems
Debu Sinha
Independent Researcher
debusinha2009@gmail.com
Abstract
Retrieval-Augmented Generation (RAG) systems remain susceptible to hallucinations despite
grounding in retrieved evidence. Current detection methods rely on semantic similarity and
natural language inference (NLI), but their fundamental limitations have not been rigorously
characterized. We apply conformal prediction to hallucination detection, providing finite-
sample coverage guarantees that enable precise quantification of detection capabilities. Using
calibration sets of n≈600examples, we achieve 94% coverage with 0% false positive rate on
synthetic hallucinations (Natural Questions). However, on three real hallucination benchmarks
spanning multiple LLMs (GPT-4, ChatGPT, GPT-3, Llama-2, Mistral), embedding-based
methods—including state-of-the-art OpenAI text-embedding-3-large and cross-encoder models—
exhibit unacceptable false positive rates: 100% on HaluEval, 88% on RAGTruth, and 50%
on WikiBio. Crucially, GPT-4 as an LLM judge achieves only 7% FPR (95% CI: [3.4%,
13.7%]) on the same data, proving the taskissolvable through reasoning. We term this
the “semantic illusion”: semantically plausible hallucinations preserve similarity to source
documents while introducing factual errors invisible to embeddings. This limitation persists
across embedding architectures, LLM generators, and task types, suggesting embedding-based
detection is insufficient for production RAG deployment.
1 Introduction
Retrieval-Augmented Generation (RAG) has become a standard approach for grounding Large
Language Models in external knowledge [ 1]. However, RAG systems still produce hallucinations—
fabricating information, misinterpreting documents, or generating unsupported claims. For applica-
tions in healthcare, legal, and financial domains, practitioners need detection methods with known
error rates.
Current detection approaches—including NLI-based consistency checks [ 2], atomic claim decom-
position [ 3], and embedding similarity [ 12]—produce uncalibrated confidence scores. A threshold of
0.5 might yield 90% recall on one dataset and 60% on another. More fundamentally, these methods
assume that hallucinations are semantically distinguishable from faithful responses, an assumption
that has not been rigorously tested.
We address this gap by applying conformal prediction to hallucination detection. Our framework,
Conformal RAG Guardrails (CRG), uses Split Conformal Prediction [ 4] to transform heuristic
scores into calibrated decisions with finite-sample coverage guarantees. This enables us to precisely
characterize when semantic-based detection succeeds and when it fails.
1arXiv:2512.15068v1  [cs.LG]  17 Dec 2025

Contributions:
1.We apply conformal prediction to RAG hallucination detection, providing finite-sample
coverage guarantees.
2.We demonstrate the “semantic illusion” across three real hallucination benchmarks and six
LLMs: embedding methods achieve 0% FPR on synthetic hallucinations but 50-100% FPR
on real LLM hallucinations (HaluEval, RAGTruth, WikiBio), while GPT-4 achieves only 7%
FPR (95% CI: [3.4%, 13.7%])—proving the limitation is modality-specific, not task-inherent.
3.We provide a methodology for rigorously evaluating any hallucination detection approach
under statistical guarantees.
4. We discuss implications for RAG deployment in high-stakes domains.
2 Related Work
Hallucination Detection.SelfCheckGPT [2] uses sampling consistency to detect hallucinations.
FActScore [ 3] decomposes responses into atomic claims for verification. G-Eval [ 8] leverages GPT-4 as
a judge. Recent work has focused specifically on RAG systems: ReDeEP [ 12] detects hallucinations
by decoupling parametric knowledge from retrieved context using mechanistic interpretability, while
HALT-RAG [ 13] uses calibrated NLI ensembles achieving strong F1 scores across tasks. LRP4RAG
[14] applies layer-wise relevance propagation to identify hallucinations. A comprehensive benchmark
[15] evaluates detection methods including LLM-as-a-Judge and HHEM across six RAG applications.
Recent surveys [ 20] position RAG and reasoning as the two dominant mitigation strategies. While
these advances improve detection accuracy, none provide calibrated thresholds with formal coverage
guarantees, nor do they rigorously characterizewhenembedding-based methods fundamentally fail.
Embedding Limitations.Recent theoretical work [ 19] demonstrates that text embeddings
capture surface-level semantics (lexical, syntactic, topical similarity) while largely neglecting implicit
meaning and factual accuracy. [ 18] proposes probabilistic hallucination detection using embedding
distances but does not address the fundamental limitation we identify. Mechanistic interpretability
work [17] shows that hallucinations arise when models over-rely on parametric memory (encoded
in FFNs) rather than retrieved context (processed by attention heads)—suggesting that semantic
similarity between response and context is orthogonal to factual grounding. Our work provides
the first rigorous empirical quantification of this limitation across multiple datasets and SOTA
embedding models.
Conformal Prediction in NLP.Conformal methods have been applied to classification [ 5],
question answering [ 6], and language model calibration [ 7]. Most relevant to our work, [ 16] applies
conformal prediction to determine when LLMs should abstain, achieving bounded hallucination
rates on open-domain QA by self-evaluating response consistency. However, their approach targets
epistemicuncertainty(“doesthemodelknow?”) inclosed-booksettings, whileweaddressfaithfulness
(“is the output supported by retrieved context?”) in RAG systems.
Calibration Methods.Temperature scaling and Platt scaling [ 9] improve probability calibra-
tion but do not provide finite-sample coverage guarantees. Conformal prediction offers distribution-
free validity.
2

3 Methodology
3.1 Problem Formulation
Letx= (q,D)be an input with query qand retrieved documents D. Letrbe the generated response.
DefineY∈{0,1}whereY= 1indicates hallucination (response not faithful toD).
We construct a detectorC α(x,r)∈{0,1}such that:
P(Y= 1 =⇒ C α(x,r) = 1)≥1−α(1)
This guarantees the False Negative Rate is bounded byα.
3.2 Nonconformity Scores
We define three scores where higher values indicate greater likelihood of hallucination:
1. Retrieval-Attribution Divergence (RAD).Using sentence encoder ϕ(BGE-base-en-
v1.5):
SRAD(x,r) = 1−max
d∈Dcos(ϕ(r),ϕ(d))(2)
2. Semantic Entailment Calibration (SEC).Using NLI model (BART-large-MNLI), for
response sentences{r j}:
SSEC(x,r) = 1−min
jmax
d∈DPNLI(entails|d,r j)(3)
3. Token-Level Factual Grounding (TFG).Fraction of content tokens with similarity below
thresholdγ= 0.3:
STFG(x,r) =1
|Tr|/summationdisplay
t∈T rI[max
t′∈TDsim(t,t′)<γ](4)
Ensemble:S(x,r) =β 1SRAD+β 2SSEC+β 3STFG(equal weights; see Appendix A)
3.3 Conformal Calibration
Given calibration set of hallucinated examples{(x i,ri)}n
i=1withYi= 1:
1. Compute scoress i=S(xi,ri)and sort:s (1)≤...≤s (n)
2. Find threshold:ˆτ=s (⌈(n+1)α⌉) (theα-quantile)
3. At test time, flag ifS(x test,rtest)≥ˆτ
By exchangeability, this guarantees marginal coverage≥1−α.
4 Experiments
4.1 Setup
Datasets.We evaluate on four benchmarks spanning synthetic and real hallucinations:
•Natural Questions (NQ-Open):Single-document QA with synthetic hallucinations (an-
swers swapped between examples). Serves as a sanity check where semantic separation is
expected.
3

•HaluEval[ 10]: Real LLM-generated hallucinations from ChatGPT, containing subtle factual
errors that preserve semantic plausibility.
•WikiBio GPT-3[ 2]: 238 GPT-3 generated biographical texts with human per-sentence
annotations (major/minor inaccurate).
•RAGTruth[ 11]: 300 examples from multiple LLMs (GPT-4, GPT-3.5, Llama-2, Mistral)
across QA, summarization, and data-to-text tasks with word-level hallucination annotations.
Data Construction.For NQ, we construct synthetic hallucinations usinganswer-swapping:
pairing each document with a semantically plausible but incorrect answer from a different question
in the same topic domain. This creates hallucinations that maintain semantic coherence with the
document while being factually incorrect—more challenging than simple refusal responses. For
HaluEval, we use the original dataset which contains real ChatGPT hallucinations paired with
correct answers, enabling evaluation on realistic errors.
Implementation.BGE-base-en-v1.5 for embeddings, BART-large-MNLI for entailment. Single
RTX 3090 on Vast.ai, total compute: <$1. Calibration: 595-629 hallucinated examples per dataset
(addressing prior sample size limitations).
4.2 Results
Table 1:CRG performance at 95% target coverage (α= 0.05).
DatasetnTarget Actual Cov. FPR Overlap Acc.
Natural Questions 595 95%95.80% 0.00%0.00%97.87%
HaluEval (ChatGPT) 629 95% 94.53% 100.00% 28.37% 47.50%
WikiBio (GPT-3) 238 95% 96.53% 50.00% 67.89% 91.60%
RAGTruth (Multi-LLM) 300 95% 94.50% 87.70% 41.40% 50.00%
Natural Questions: Framework Validation.CRG achieves 95.80% coverage with 0% FPR
on synthetic hallucinations ( n= 595), demonstrating that conformal calibration produces valid
guarantees when the task has clear semantic separation. The score distributions show 0% overlap
(Figure 1), with faithful responses clustering near 0.11 and hallucinated responses near 0.49.
HaluEval: The Real Hallucination Challenge.On real LLM-generated hallucinations
(n= 629), CRG achieves near-target coverage (94.53%) but with 100% FPR. This result reveals
thatreal hallucinations are semantically indistinguishable from faithful responses using embedding
and NLI-based methods. The score distributions show 28.37% overlap, explaining why any threshold
that achieves coverage on hallucinations also flags faithful responses.
WikiBio: A Contrasting Case.The WikiBio GPT-3 dataset ( n= 238) reveals a different
failure mode. Unlike HaluEval, embeddings achieve 50% FPR—better than random but still
inadequate for production. Critically, the score direction isreversed: hallucinations scorehigher
(µhal= 0.162) than faithful responses ( µfaith = 0.066), opposite to HaluEval ( µhal= 0.332vs
µfaith= 0.302). This suggests GPT-3’s gross factual errors (per-sentence human annotations) create
detectable semantic divergence, while ChatGPT’s hallucinations maintain semantic plausibility—
precisely the “semantic illusion” phenomenon. The high accuracy (91.6%) is misleading due to class
imbalance (92% hallucinated).
RAGTruth: Cross-Model Validation.RAGTruth ( n= 300) provides the most compre-
hensive test, spanning six LLMs (GPT-4, GPT-3.5-turbo, Llama-2-7B/13B/70B, Mistral-7B) and
4

three task types (QA, summarization, data-to-text). With 87.7% FPR and 50% accuracy (random
chance), the results confirm that embedding failure ismodel-agnostic: whether hallucinations come
from GPT-4 or Llama-2, embeddings cannot distinguish them from faithful responses. Notably,
FPR remains high across all model families (OpenAI: 89%, Llama-2: 86%, Mistral: 88%) and task
types (QA: 85%, Summary: 90%, Data2txt: 87%), indicating no subset where embeddings succeed.
The 41.4% score overlap explains this failure—both classes occupy the same embedding space region.
4.3 Ablation: Sensitivity to Coverage Target
We varyαfrom 0.01 to 0.20 to examine whether relaxing coverage requirements improves FPR on
real hallucinations.
Table 2:Ablation on coverage targetα(lowerα= higher coverage target).
NQ (Synthetic) HaluEval (Real)
αCov. FPR Acc. Cov. FPR Acc.
0.01 98.5%0%99.2% 99.0% 100% 49.5%
0.05 94.3%0%97.1% 94.3% 100% 47.4%
0.10 91.6%0%95.8% 90.6% 99.5% 45.6%
0.20 81.0%0%90.4% 80.6% 90.2% 45.3%
Key Finding:On NQ, FPR remains 0% across all αvalues—the score distributions are
completely separable. On HaluEval, even at α= 0.20(80% coverage), FPR remains at 90%. This
confirms the failure is not threshold-dependent.
4.4 Individual Score Ablation
To verify that the failure is not specific to our ensemble, we evaluate each score component
individually on HaluEval at 95% target coverage.
Table 3:Individual score component ablation on HaluEval (95% target coverage). All semantic-based scores
fail regardless of implementation.
Score Method FPR Coverage
RAD Embedding cosine similarity 100% 94.5%
SEC NLI entailment probability 100% 94.2%
TFG Token-level grounding 98% 95.1%
Ensemble Weighted average 100% 94.5%
Key Finding:All three individual scores fail with ≥98% FPR because they all measure
semantic similarity in different ways. RAD captures document-response embedding distance, SEC
measures entailment probability, and TFG assesses token-level lexical overlap. The consistent failure
across these diverse approaches suggests the problem is fundamental to semantic similarity, not
specific to any particular implementation.
5

4.5 Baseline Comparison
We compare CRG against a SelfCheckGPT-style baseline using NLI consistency scoring. For fair
comparison, we calibrate the baseline threshold to match CRG’s target coverage.
Table 4:CRG vs SelfCheckGPT baseline at matched coverage.
Dataset Method Coverage FPR Accuracy
NQ (Synthetic)CRG 95.80%0.00%97.87%
Baseline 99.51%0.00%99.75%
HaluEval (Real)CRG 94.53% 100.00% 47.50%
Baseline 94.53% 100.00% 47.50%
On synthetic hallucinations, both methods perform well with 0% FPR—the score distributions
are completely separable. On HaluEval,both methods fail identically, achieving random-chance
accuracy (47.50%) despite near-target coverage. This reveals a fundamental limitation: semantic
similarity cannot distinguish factually incorrect responses that preserve semantic plausibility.
4.6 SOTA Detection Methods Comparison
A natural objection is that embedding failure might be due to model capacity or approach. We test
state-of-the-art detection methods beyond basic embeddings:
•HALT-RAG Style: Calibrated NLI ensemble using DeBERTa-v3 + BART-MNLI, follow-
ing [13]
•ReDeEP Style: Attention pattern analysis measuring response-to-document attention vs
self-attention, approximating mechanistic interpretability [12]
•OpenAI Embeddings: text-embedding-3-large (3072 dimensions)
•Cross-Encoder: BGE-Reranker-v2-m3
Table 5:SOTA detection methods on HaluEval at 95% target coverage. 95% Wilson CIs shown.All
semantic-based methods fail; only LLM reasoning succeeds.
Method Coverage FPR FPR 95% CI Accuracy
Basic Embeddings
BGE-base-en-v1.5 94.3% 100% [99.0%, 100%] 47.4%
OpenAI text-embedding-3-large 94.9% 100% [96.3%, 100%] 46.0%
Advanced Semantic Methods
BGE-Reranker-v2-m3 (Cross-Enc.) 94.9% 99% [94.6%, 99.8%] 46.5%
HALT-RAG Style (NLI Ensemble) 93.2% 57% [46.0%, 67.6%] 67.3%
ReDeEP Style (Attention) 98.6% 100% [95.2%, 100%] 48.0%
Reasoning-Based
GPT-4o-mini Judge 67.0%7% [3.4%, 13.7%] 80.0%
6

Critical Finding:The results reveal a striking pattern:all semantic-based detection methods
fail, regardless of sophistication. Basic embeddings (BGE, OpenAI) achieve 100% FPR. Advanced
methods show marginal improvement: HALT-RAG style NLI ensemble achieves 57% FPR (still
unacceptable for production), while ReDeEP-style attention analysis achieves 100% FPR. Only
reasoning-based detection (GPT-4 Judge) succeeds with 7% FPR. This demonstrates that the
“semantic illusion” is not a limitation of specific embedding architectures but afundamental property
of semantic similarity measureswhen applied to RLHF-trained model outputs.
Fair Comparison at Matched Coverage.A potential concern is that GPT-4’s lower FPR
might be due to its lower coverage (67% vs 95%). To address this, we estimate CRG’s FPR at 67%
coverage by varying the threshold. At α= 0.33(67% target coverage), embeddings still achieve
approximately 78% FPR—demonstrating that the 71 percentage point gap vs GPT-4’s 7% FPR
isnotan artifact of coverage mismatch but reflects a fundamental limitation of semantic-based
detection.
4.7 LLM-as-Judge: The Gold Standard
To verify HaluEval hallucinations are genuinely detectable, we test GPT-4o-mini as a judge (prompt
and methodology in Appendix A). GPT-4 achieves 7% FPR (95% CI: [3.4%, 13.7%]) compared to
100% for embeddings—proving the taskissolvable through reasoning about factual consistency.
The signal exists; embeddings simply cannot access it.1
4.8 Score Distribution Analysis
Table 6:Score statistics and calibration thresholds.
Datasetnˆτ µ hal µfaith Overlap
NQ (Synthetic) 595 0.474 0.493±0.02 0.110±0.03 0.00%
HaluEval (Real) 629 0.060 0.332±0.14 0.302±0.12 28.37%
The contrast is striking. On NQ, hallucinated responses ( µ= 0.493) are clearly separable from
faithful responses ( µ= 0.110), with 0% distribution overlap. On HaluEval, the distributions nearly
coincide (µhal= 0.332vsµfaith= 0.302) with 28.37% overlap. The calibration threshold drops from
0.474 to 0.060, but this low threshold captures the lower tail of both distributions indiscriminately.
5 Discussion
5.1 Why Semantic Methods Fail on Real Hallucinations
The contrast between NQ (0% FPR) and HaluEval (100% FPR) reveals what we term the “semantic
illusion”: real LLM hallucinations preserve semantic plausibility while introducing subtle factual
errors. Synthetic hallucinations, created by swapping answers between examples, are semantically
distant from the retrieved context and trivially detectable. Real hallucinations maintain the stylistic
patterns, topical relevance, and syntactic structure of faithful responses—they differ only in factual
accuracy.
1GPT-4’s 67% coverage (vs 95% target) reflects that the LLM judge was not conformally calibrated—we report its
raw classification performance. Even without calibration, its 7% FPR vastly outperforms embeddings’ 100% FPR,
demonstrating the fundamental modality gap.
7

Figure 1:Score distributions for NQ (left) and HaluEval (right). NQ shows complete separation between
faithful (green) and hallucinated (red) responses, enabling effective detection. HaluEval shows substantial
overlap, explaining the 100% FPR.
Figure 1 visualizes this phenomenon. The NQ distributions show complete separation (0% over-
lap), with a clear decision boundary at τ= 0.474. The HaluEval distributions overlap substantially
(28.37%), with both faithful and hallucinated responses spanning the same score range. No threshold
can achieve both high coverage and low FPR under such overlap.
The WikiBio Anomaly and Hallucination Taxonomy.WikiBio’s reversed score direction
(hallucinations scorehigherthan faithful responses) reveals a hallucination taxonomy with important
implications:
•Type 1: Confabulations(GPT-3 style, WikiBio): Factually incorrect statements that
diverge semantically from source material. These introduce unrelated content or make up
facts wholesale, creating detectable semantic distance. The pre-RLHF GPT-3 model produces
these errors.
•Type 2: Plausible Paraphrases(ChatGPT/GPT-4 style, HaluEval): Factually incorrect
statements that preserve semantic similarity. RLHF optimization produces fluent, topically
relevant responses that introduce subtle factual errors while maintaining stylistic coherence
with the source.
This distinction has a critical implication:as models improve through RLHF alignment,
embedding-based detection becomes less effective. Better language models produce more semanti-
cally plausible hallucinations that are invisible to semantic similarity measures. This suggests a
fundamental tension between model capability and detection feasibility.
Error Analysis: Validating the Taxonomy.We manually analyzed 30 examples from
each dataset to validate this taxonomy. On WikiBio (GPT-3), 73% of hallucinations involved
fabricating biographical facts absent from the source (e.g., inventing awards, incorrect dates)—these
create detectable semantic divergence. On HaluEval (ChatGPT), 87% of hallucinations were subtle
rewordings or logical inferences that plausibly follow from the source but introduce factual errors
(e.g., misattributing claims, overgeneralizing statements). This error analysis confirms the Type
1/Type 2 distinction: GPT-3 produces detectable confabulations while ChatGPT produces plausible
paraphrases.
5.2 Implications
These results have several implications for practitioners:
8

1.Detection methods relying on embedding similarity or NLI may not generalize from synthetic
to real hallucinations.
2.High performance on constructed benchmarks (e.g., answer-swapping) may not predict pro-
duction performance.
3.Effective detection likely requires scores that capture factual accuracy rather than semantic
similarity, such as knowledge graph grounding or specialized fact verification models.
5.3 Utility of the Conformal Framework
The conformal prediction framework remains useful for evaluating detection methods:
1. Any proposed nonconformity score can be evaluated under the same methodology.
2.When a score does provide separation, conformal prediction yields finite-sample coverage
guarantees.
3. Users can specify desired coverage (1−α) rather than selecting arbitrary thresholds.
Important Caveat:Conformal guarantees are only practically useful when score distributions
are separable. When FPR approaches 100% (as on HaluEval), the coverage guarantee is technically
satisfied but operationally meaningless—flagging all responses achieves high coverage but provides
no discrimination. Our results demonstrate precisely when this occurs: embeddings provide
useful guarantees on synthetic hallucinations (0% FPR) but not on semantically plausible real
hallucinations.
5.4 Limitations and Future Work
Our study has several limitations. While we use calibration sets of n≈300-600, larger samples
would further stabilize threshold estimates. The variation in FPR across datasets (100% HaluEval,
88% RAGTruth, 50% WikiBio) suggests hallucination detectability varies by generation model and
error type—GPT-3’s gross factual errors are more detectable than ChatGPT’s semantically plausible
hallucinations. Future work should characterize this spectrum more precisely. Most importantly, our
negative result motivates the search for nonconformity scores that capture factual accuracy rather
than semantic similarity—potential directions include knowledge graph grounding, chain-of-thought
verification, or attribution methods [14].
6 Conclusion
We applied conformal prediction to RAG hallucination detection, demonstrating the “semantic
illusion” across three real hallucination benchmarks and six LLMs. Embedding-based methods
achieve 0% FPR on synthetic hallucinations but 100% FPR on HaluEval (ChatGPT), 88% on
RAGTruth (GPT-4, Llama-2, Mistral), and 50% on WikiBio (GPT-3)—even for SOTA embeddings
like OpenAI text-embedding-3-large and cross-encoders. The failure is model-agnostic: whether
hallucinations come from GPT-4 or Llama-2-7B, embeddings cannot distinguish them from faithful
responses.
The critical finding is that GPT-4 achieves 7% FPR (95% CI: [3.4%, 13.7%]) on the same
HaluEval data where embeddings fail completely. This proves the task is solvable through reasoning;
embeddings simply cannot access the relevant signal. Ablation across coverage targets confirms this
9

is not threshold-dependent: on real hallucinations, FPR stays above 90% regardless of operating
point.
Our results suggest that simple semantic similarity measures (embedding cosine similarity,
NLI entailment) are insufficient for detecting semantically plausible hallucinations from RLHF-
trained models. Production RAG systems should consider reasoning-based verification or specialized
detection methods (e.g., mechanistic interpretability, knowledge graph grounding) rather than
relying solely on basic embedding-based guardrails. Future work should evaluate whether SOTA
detection approaches provide better nonconformity scores for conformal frameworks.
Reproducibility
Code available at https://github.com/debu-sinha/conformal-rag-guardrails . Embedding-
based experiments: single RTX 3090, <$1 compute. LLM-as-Judge baseline (GPT-4o-mini): $0.02
for 200 examples via OpenAI API ( $0.10 per 1K queries at production scale).
References
[1]P. Lewis et al. Retrieval-augmented generation for knowledge-intensive NLP tasks.NeurIPS,
2020.
[2]P. Manakul et al. SelfCheckGPT: Zero-resource black-box hallucination detection.EMNLP,
2023.
[3] S. Min et al. FActScore: Fine-grained atomic evaluation of factual precision.EMNLP, 2023.
[4] V. Vovk et al.Algorithmic Learning in a Random World. Springer, 2005.
[5] S. Ravfogel et al. Conformal nucleus sampling.Findings of ACL, 2023.
[6] V. Quach et al. Conformal language modeling.ICLR, 2024.
[7] B. Kumar et al. Conformal prediction with large language models.arXiv:2305.18404, 2023.
[8]Y. Liu et al. G-Eval: NLG evaluation using GPT-4 with better human alignment.EMNLP,
2023.
[9] C. Guo et al. On calibration of modern neural networks.ICML, 2017.
[10] J. Li et al. HaluEval: A large-scale hallucination evaluation benchmark.EMNLP, 2023.
[11]Y. Wang et al. RAGTruth: A hallucination corpus for developing trustworthy retrieval-
augmented language models.arXiv, 2023.
[12]Y. Xiang et al. ReDeEP: Detecting hallucination in retrieval-augmented generation via mecha-
nistic interpretability.ICLR, 2025.
[13]Z. Wang et al. HALT-RAG: A task-adaptable framework for hallucination detection with
calibrated NLI ensembles.arXiv:2509.07475, 2025.
[14] H. Hu et al. LRP4RAG: Detecting hallucinations in retrieval-augmented generation via layer-
wise relevance propagation.AAAI, 2025.
10

[15]Z. Fang et al. Real-time evaluation models for RAG: Who detects hallucinations best?
arXiv:2503.21157, 2025.
[16]Y. Abbasi-Yadkori, et al. Mitigating LLM hallucinations via conformal abstention.arXiv
preprint arXiv:2405.01563, 2024.
[17]X. Chen et al. Detecting hallucinations in graph retrieval-augmented generation via attention
patterns and semantic alignment.arXiv:2512.09148, 2025.
[18]L. Zhang et al. Hallucination detection: A probabilistic framework using embeddings distance
analysis.arXiv:2502.08663, 2025.
[19]W. Li et al. Text embeddings should capture implicit semantics, not just surface meaning.
arXiv:2506.08354, 2025.
[20]S. Wang et al. Mitigating hallucination in large language models: An application-oriented
survey on RAG, reasoning, and agentic systems.arXiv:2510.24476, 2025.
A Implementation Details
A.1 Ensemble Weights
The nonconformity score ensemble uses equal weights: β1=β2=β3= 1/3for RAD, SEC, and
TFG respectively. We found results robust to weight variations in preliminary experiments.
A.2 GPT-4 Judge Methodology
For the LLM-as-Judge baseline, we use GPT-4o-mini via OpenAI API with the following zero-shot
prompt:
You are a fact-checking assistant. Given a source
document and a response, determine if the response
is faithful to the document or contains hallucinations.
Document: {document}
Response: {response}
Is this response faithful to the document?
Answer only "faithful" or "hallucination".
API parameters: temperature=0, max_tokens=10. We classify “hallucination” responses as
positive predictions. Total API cost for 200 examples: approximately $0.02.
A.3 Confidence Intervals
For FPR estimates, we compute 95% Wilson score intervals. With n= 200test examples and
observed FPR of 100%, the 95% CI is [98.2%, 100%]. For RAGTruth (87.7% FPR, n= 65faithful
test examples), the 95% CI is [77.2%, 94.5%]. These intervals confirm the statistical significance of
the embedding failure.
11

B Towards Alternative Approaches
Our negative result motivates exploration of detection methods beyond semantic similarity:
Knowledge Graph Grounding.Rather than embedding similarity, verify claims against
structured knowledge bases. Recent work [ 17] shows attention patterns differ between grounded
and hallucinated responses.
Chain-of-Thought Verification.Decompose responses into atomic claims and verify each via
reasoning, as in FActScore [ 3]. Our GPT-4 baseline (11% FPR) suggests reasoning-based approaches
can succeed where embeddings fail.
Hybrid Architectures.Combine efficient embedding-based pre-filtering with expensive LLM
verification for flagged responses. This could achieve production-viable latency while maintaining
accuracy.
Distillation from LLM Judges.Train smaller, specialized models on GPT-4 judgment data
to achieve similar accuracy at lower inference cost. This transfers reasoning capability into an
efficient classifier.
We leave empirical comparison of these approaches to future work, noting that our conformal
framework provides a principled evaluation methodology for any proposed nonconformity score.
12