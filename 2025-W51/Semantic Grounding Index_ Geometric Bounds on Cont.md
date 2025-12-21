# Semantic Grounding Index: Geometric Bounds on Context Engagement in RAG Systems

**Authors**: Javier Marín

**Published**: 2025-12-15 18:09:54

**PDF URL**: [https://arxiv.org/pdf/2512.13771v1](https://arxiv.org/pdf/2512.13771v1)

## Abstract
When retrieval-augmented generation (RAG) systems hallucinate, what geometric trace does this leave in embedding space? We introduce the Semantic Grounding Index (SGI), defined as the ratio of angular distances from the response to the question versus the context on the unit hypersphere $\mathbb{S}^{d-1}$.Our central finding is \emph{semantic laziness}: hallucinated responses remain angularly proximate to questions rather than departing toward retrieved contexts. On HaluEval ($n$=5,000), we observe large effect sizes (Cohen's $d$ ranging from 0.92 to 1.28) across five embedding models with mean cross-model correlation $r$=0.85. Crucially, we derive from the spherical triangle inequality that SGI's discriminative power should increase with question-context angular separation $θ(q,c)$-a theoretical prediction confirmed empirically: effect size rises monotonically from $d$=0.61 -low $θ(q,c)$, to $d$=1.27 -high $θ(q,c)$, with AUC improving from 0.72 to 0.83. Subgroup analysis reveals that SGI excels on long responses ($d$=2.05) and short questions ($d$=1.22), while remaining robust across context lengths. Calibration analysis yields ECE=0.10, indicating SGI scores can serve as probability estimates, not merely rankings. A critical negative result on TruthfulQA (AUC=0.478) establishes that angular geometry measures topical engagement rather than factual accuracy. SGI provides computationally efficient, theoretically grounded infrastructure for identifying responses that warrant verification in production RAG deployments.

## Full Text


<!-- PDF content starts -->

SEMANTICGROUNDINGINDEX: GEOMETRICBOUNDS ON
CONTEXTENGAGEMENT INRAG SYSTEMS
Javier Marin
CERT
javier@jmarin.info
December 17, 2025
ABSTRACT
When retrieval-augmented generation (RAG) systems hallucinate, what geometric trace does this
leave in embedding space? We introduce the Semantic Grounding Index (SGI), defined as the ratio
of angular distances from the response to the question versus the context on the unit hypersphere
Sd−1. Our central finding issemantic laziness: hallucinated responses remain angularly proximate
to questions rather than departing toward retrieved contexts. On HaluEval (n= 5,000), we ob-
serve large effect sizes (Cohen’sdranging from0.92to1.28) across five embedding models with
mean cross-model correlationr= 0.85. Crucially, we derive from the spherical triangle inequality
that SGI’s discriminative power should increase with question-context angular separationθ(q, c)—a
theoretical prediction confirmed empirically: effect size rises monotonically fromd= 0.61(low
θ(q, c)) tod= 1.27(highθ(q, c)), with AUC improving from0.72to0.83. Subgroup analysis re-
veals that SGI excels on long responses (d= 2.05) and short questions (d= 1.22), while remaining
robust across context lengths. Calibration analysis yields ECE= 0.10, indicating SGI scores can
serve as probability estimates, not merely rankings. A critical negative result on TruthfulQA (AUC
= 0.478) establishes that angular geometry measures topical engagement rather than factual accu-
racy. SGI provides computationally efficient, theoretically grounded infrastructure for identifying
responses that warrant verification in production RAG deployments.
1 Introduction
LLMs generate text through autoregressive next-token prediction, optimizing for distributional statistics of training
corpora (Brown et al., 2020; Touvron et al., 2023). This objective produces fluent continuations without maintaining
explicit correspondence to external reality, a characteristic that manifests ashallucination(Ji et al., 2023; Huang et
al., 2023). Retrieval-Augmented Generation (RAG) architectures condition generation on retrieved documents (Lewis
et al., 2020; Guu et al., 2020), yet hallucination persists: models fabricate claims absent from context or fail to
substantively engage with retrieved information (Shuster et al., 2021; Bao et al., 2025; Niu et al., 2024).
We investigate a geometric question: when a RAG system fails to ground its response in the provided context, what
signature does this leave in embedding space? Modern sentence transformers are trained via contrastive objectives
that explicitly optimize angular relationships on the unit hypersphere (Reimers and Gurevych, 2019; Wang and Isola,
2020). This makesSd−1the natural geometric setting for analyzing response behavior.
Our central contribution is the identification, theoretical characterization, and empirical validation of a geometric pat-
tern we termsemantic laziness. When models hallucinate in RAG systems, their responses remain angularly proximate
to the question rather than departing toward the context’s semantic territory. We formalize this through the Seman-
tic Grounding Index (SGI)—the ratio of angular distancesθ(r, q)/θ(r, c)—and demonstrate that it provides a robust,
theoretically grounded signal for detecting context disengagement.
Contributions.arXiv:2512.13771v1  [cs.AI]  15 Dec 2025

APREPRINT- DECEMBER17, 2025
1. We introduce SGI as an intrinsic quantity onSd−1and derive geometric bounds from the spherical triangle
inequality thatpredictwhen discrimination will be most effective.
2. Weconfirm this theoretical prediction empirically: effect size increases monotonically with question-context
angular separation (d= 0.61→0.90→1.27acrossθ(q, c)terciles).
3. We establish cross-model robustness at scale (n= 5,000): five architecturally distinct embedding models
correlate atr= 0.85with ranking agreementρ= 0.87.
4. We characterize operational boundaries: SGI excels on long responses and short questions, maintains calibra-
tion (ECE= 0.10), and fails predictably on TruthfulQA where angular geometry cannot discriminate factual
accuracy.
2 Theoretical Foundations
2.1 The Embedding Hypersphere
Contrastive learning objectives for sentence embeddings decompose into alignment and uniformity terms on the unit
hypersphere (Wang and Isola, 2020). The InfoNCE loss encourages matched pairs to cluster while spreading unrelated
points apart, inducing structure onSd−1where L2-normalized embeddings reside.
Letϕ:S →Rddenote a sentence embedding model and ˆϕ(s) =ϕ(s)/∥ϕ(s)∥the L2-normalized representation. The
normalized embeddings lie on:
Sd−1={x∈Rd:∥x∥= 1}(1)
This is a compact Riemannian manifold with constant positive curvature (Mardia and Jupp, 2000). The intrinsic
distance is the geodesic (great-circle arc length):
θ(a, b) = arccos( ˆϕ(a)⊤ˆϕ(b))(2)
This angular distanceθ∈[0, π]satisfies all metric axioms onSd−1, including the triangle inequality (Bridson and
Haefliger, 2013). We note that while cosine similarity is ubiquitous in applications, it does not satisfy the triangle
inequality; angular distance is the proper metric for geometric analysis (You, 2025).
2.2 The Semantic Grounding Index
For a RAG instance(q, c, r)with questionq, retrieved contextc, and generated responser, we define the Semantic
Grounding Index as:
SGI(r;q, c) =θ(r, q)
θ(r, c)=arccos( ˆϕ(r)⊤ˆϕ(q))
arccos( ˆϕ(r)⊤ˆϕ(c))(3)
Equation 3 measures the ratio of angular departures, how far the response has traveled from the question relative to its
distance from the context. When SGI>1, the response is angularly farther from the question than from the context—
it has “departed” toward the context’s semantic territory. When SGI<1, the response remains closer to the question
than to the context.
2.3 Geometric Bounds and Theoretical Predictions
The spherical triangle inequality constrains admissible SGI values. For anyq, c, r∈Sd−1:
|θ(q, c)−θ(r, c)| ≤θ(r, q)≤θ(q, c) +θ(r, c)(4)
Dividing byθ(r, c)yields bounds on SGI:θ(q, c)
θ(r, c)−1≤SGI≤θ(q, c)
θ(r, c)+ 1(5)
These bounds form the basis of our theoretical contribution:
SGI’s discriminative power should increase withθ(q, c). When question and context are seman-
tically similar (small (θ(q, c))), the triangle inequality constrains SGI values near 1 regardless of
response quality. Whenθ(q, c)is large, the constraint relaxes, permitting greater separation be-
tween grounded and ungrounded responses.
2

APREPRINT- DECEMBER17, 2025
θ(q, c)
θ(r, q)θ(r, c)
θ(r, q)θ(r, c)
qc
rvalid
rhallucSd−1
Valid response:
Largeθ(r, q), smallθ(r, c)
SGI=θ(r, q)
θ(r, c)>1
Hallucination:
Smallθ(r, q), largeθ(r, c)
SGI=θ(r, q)
θ(r, c)<1
Figure 1: Angular geometry of SGI on the unit hypersphere. Questionqand contextcdefine anchor points; their
angular separationθ(q, c)determines the geometric “room” for response differentiation. A valid response (blue)
departs fromqtowardc, yielding SGI>1. A hallucination (red) remains angularly proximate to the question—the
semantic laziness signature—yielding SGI<1.
This is the mathematical consequence of the triangle inequality. If SGI captures something real about semantic ground-
ing, we should observe effect sizes that increase monotonically withθ(q, c). We test this prediction explicitly in
Section 5.3.
2.4 The Semantic Laziness Hypothesis
We hypothesize that hallucinated responses in RAG systems exhibitsemantic laziness: rather than introducing vo-
cabulary and concepts from the retrieved context, they produce completions that remain in the question’s semantic
neighborhood.
LetR validandR halluc denote distributions over valid and hallucinated responses. The semantic laziness hypothesis
predicts:
Er∼R valid[SGI(r;q, c)]>E r∼R halluc[SGI(r;q, c)](6)
This hypothesis connects to how autoregressive models handle uncertainty. When a model lacks confidence in how to
use retrieved context, it may default to “safe” completions that echo the question’s framing rather than venturing into
the context’s semantic territory.
3 Related Work
3.1 Geometric Methods for Hallucination Detection
Li et al. (2025) compute semantic volume from batches of responses to quantify uncertainty. Catak et al. (2024) apply
convex hull analysis to embedding spaces. Gao et al. (2025) found that hallucinated responses exhibit smaller devia-
tions from prompts in hidden state space—an observation consistent with our semantic laziness characterization. Our
work differs by focusing on the triangular geometry of question-context-response relationships, deriving theoretical
bounds, and establishing cross-model robustness.
3.2 Semantic Entropy and Consistency Methods
Farquhar et al. (2024) introduced semantic entropy for hallucination detection via multiple sampling. Kuhn et al.
(2023) developed linguistic invariances for uncertainty estimation. These methods require multiple generation passes;
SGI operates on single responses.
3

APREPRINT- DECEMBER17, 2025
3.3 NLI-Based Detection
SummaC (Laban et al., 2022), HALT-RAG (HALT-RAG, 2025), and LettuceDetect (Kovács and Recski, 2025) frame
detection as entailment classification. These methods detect logical contradiction; SGI detects semantic disengage-
ment. The signals are complementary.
3.4 Spherical Geometry in Representation Learning
The unit hypersphere is well-studied in directional statistics (Mardia and Jupp, 2000; Fisher, 1953). Wang and Isola
(2020) analyzed contrastive learning through alignment and uniformity onSd−1. Meng et al. (2019) developed spheri-
cal text embeddings. You (2025) provided comprehensive analysis of when cosine similarity succeeds and fails, noting
that angular distance—unlike cosine similarity—satisfies the triangle inequality.
4 Experimental Design
4.1 Implementation Details
Text Preprocessing.We use spaCy (Honnibal et al., 2020) with theen_core_web_smpipeline for sentence bound-
ary detection and basic tokenization when segmenting long contexts. This lightweight model (12MB) provides suffi-
cient accuracy for our preprocessing needs without introducing computational overhead. Alternative pipelines include
en_core_web_md(40MB) anden_core_web_lg(560MB), which incorporate word vectors but offer no advantage
for boundary detection. For purely rule-based segmentation, spaCy’ssentencizercomponent or NLTK’spunktto-
kenizer are viable alternatives; we observed no significant difference in downstream SGI scores across these choices,
suggesting robustness to preprocessing variations.
Embedding Computation.We compute sentence embeddings using thesentence-transformerslibrary (v2.2.2)
(Reimers and Gurevych, 2019). For each RAG instance(q, c, r), we encode the question, context, and response as
separate strings without additional prompting or instruction prefixes. Embeddings are L2-normalized to unit length
before angular distance computation.
SGI Computation.Algorithm 1 specifies the complete procedure. Angular distances are computed viaθ(a, b) =
arccos(clip(a⊤b,−1,1)), where clipping prevents numerical errors from domain violations. We addϵ= 10−8to the
denominator to avoid division by zero whenθ(r, c)≈0.
Algorithm 1Semantic Grounding Index Computation
Require:Questionq, Contextc, Responser, Embedding modelϕ
Ensure:SGI score
1:q←ϕ(q)/∥ϕ(q)∥{L2 normalize}
2:c←ϕ(c)/∥ϕ(c)∥
3:r←ϕ(r)/∥ϕ(r)∥
4:θrq←arccos(clip(r⊤q,−1,1))
5:θrc←arccos(clip(r⊤c,−1,1))
6:returnθ rq/(θrc+ϵ)
Sampling and Splits.From HaluEval QA (10,000samples), we randomly samplen= 5,000instances stratified by
hallucination label. For TruthfulQA, we use all 817 questions, constructing paired samples by treating each question’s
correct and incorrect answers as separate instances (n= 800after filtering incomplete entries). No train/test split is
applied; we report descriptive statistics on the full samples.
Statistical Analysis.Effect sizes use Cohen’sdwith pooled standard deviation. Group comparisons use Welch’s
t-test (unequal variances). Cross-model correlations use Pearsonrfor linear agreement and Spearmanρfor rank
agreement. Calibration analysis uses expected calibration error (ECE) with 10 equal-frequency bins.
4

APREPRINT- DECEMBER17, 2025
4.2 Datasets
HaluEval QA(Li et al., 2023) provides question-knowledge-answer triples with hallucination labels. The knowl-
edge field serves as retrieved context. We usen= 5,000samples for comprehensive analysis, enabling stratified
evaluation with sufficient statistical power.
TruthfulQA(Lin et al., 2022) contains 817 questions targeting common misconceptions, with truthful and false
answers. We constructn= 800paired samples to test whether angular geometry can discriminate factual accuracy.
4.3 Embedding Models
A critical question is whether SGI measures a property of the text or an artifact of a particular embedding model. We
evaluate five sentence transformers with distinct architectures and training regimes:
•all-mpnet-base-v2(768d): General-purpose, contrastive training (Reimers and Gurevych, 2019)
•all-MiniLM-L6-v2(384d): Knowledge-distilled from larger models (Wang et al., 2022)
•bge-base-en-v1.5(768d): BAAI’s contrastive model (Xiao et al., 2023)
•e5-base-v2(768d): Microsoft’s weakly-supervised embeddings (Wang et al., 2024)
•gte-base(768d): Alibaba’s multi-stage contrastive model (Li et al., 2023b)
If SGI captures something fundamental about text, scores should correlate strongly across these models despite their
different training objectives and architectures.
4.4 Evaluation Metrics
We compute Cohen’sdeffect sizes and Welch’st-test for group comparisons. Classification performance uses ROC-
AUC. For cross-model validation, we compute Pearson correlation (linear agreement), Spearmanρ(ranking agree-
ment), and expected calibration error (ECE) for probability estimation quality.
5 Results
5.1 Cross-Model Validation on HaluEval
Table 1: Effect sizes and classification performance across five embedding models on HaluEval (n= 5,000). All
models show significant separation with large effect sizes, demonstrating that SGI captures a property of the text
rather than an embedding artifact.
Model SGI (Valid) SGI (Halluc) Cohen’sdAUCp-value
mpnet 1.142 0.921+0.920.776<0.01
minilm 1.203 0.856+1.280.824<0.01
bge 1.231 0.948+1.270.823<0.01
e5 1.138 0.912+1.030.794<0.01
gte 1.224 0.927+1.130.811<0.01
Mean1.188 0.913+1.130.806 —
Table 1 presents the primary result. Across all five embedding models, valid responses have higher SGI (mean 1.19)
than hallucinations (mean 0.91), confirming the semantic laziness hypothesis. Effect sizes range fromd= 0.92to
d= 1.28, all conventionally “large.” Withn= 5,000, allp-values are below10−50, and AUC values range from 0.78
to 0.82.
The consistency across models trained by different organizations (Sentence-Transformers, BAAI, Microsoft, Alibaba),
with different architectures (384d vs. 768d), and different training objectives (contrastive, instruction-tuned, distilled)
provides strong evidence that SGI measures a property of the text itself.
5

APREPRINT- DECEMBER17, 2025
(a) Pearson correlation matrix for SGI scores across embedding
models. Mean off-diagonal correlation:r= 0.85.
(b) Cross-model distributions.
Figure 2: Cross-model agreement for SGI scores on HaluEval (n= 5,000). High correlations across architecturally
distinct embedding models indicate that SGI captures an intrinsic property of text rather than an artifact of any partic-
ular embedding space.
5.2 Cross-Model Correlation Analysis
Figure 2 shows the cross-model correlation structure. The Pearson correlation matrix reveals mean pairwise correlation
r= 0.85, with minimumr= 0.80(mpnet–bge) and maximumr= 0.95(bge–gte). Spearman rank correlations are
comparably high (meanρ= 0.87), indicating that models agree not just on absolute SGI values but on therankingof
which samples are most and least grounded.
The bge–gte correlation ofr= 0.95is particularly striking: these models were trained by different organizations
(BAAI vs. Alibaba) with different training procedures, yet they “see” nearly identical semantic laziness behaviors.
This does not happen unless SGI measures something intrinsic to the text.
5.3 Confirming the Theoretical Prediction: Stratified Analysis
Table 2: Stratified analysis by question-context angular separationθ(q, c). Effect size increases monotonically with
θ(q, c), confirming the theoretical prediction derived from the triangle inequality.
θ(q, c)Tercilen θ(q, c)Range SGI (Valid) SGI (Halluc) Cohen’sdAUC
Low 1,667[0.42,0.89]1.08 0.94+0.610.721
Medium 1,666[0.89,1.12]1.19 0.91+0.900.768
High 1,667[1.12,1.57]1.31 0.88+1.270.832
Table 2 and Figure 3 present the critical result: effect size increases monotonically withθ(q, c). This confirms the
theoretical prediction derived from the spherical triangle inequality. The bounds in Equation 5 predict that when
θ(q, c)is small, SGI values are geometrically constrained near≈1. Whenθ(q, c)is large, the constraint relaxes,
allowing greater separation between valid and hallucinated responses.
The monotonic increase transforms SGI from “useful heuristic” to “principled method with predictable behavior.”
Practitioners can assess expected discriminative power by measuringθ(q, c)before deployment.
5.4 Subgroup Robustness: Where Does SGI Excel and Fail?
Table 3 and Figure 4 allows some interesting conclusions:
1. Response length is critical. Effect size increases fromd= 0.95(short) tod= 2.05(long). Longer responses
provide more “signal” for embedding estimation—the response vector is a more stable representation of
6

APREPRINT- DECEMBER17, 2025
Figure 3: Effect size and AUC increase monotonically with question-context angular separationθ(q, c), confirming
the theoretical prediction. Whenθ(q, c)is small, the triangle inequality constrains SGI near 1 regardless of response
quality. Whenθ(q, c)is large, there is geometric “room” for discrimination.
Table 3: Subgroup analysis by text characteristics. SGI effect size varies substantially with response length (strongest
on long responses) and question length (strongest on short questions), while remaining stable across context lengths.
Feature LevelnCohen’sdAUC
Question LengthShort 1,667+1.220.812
Medium 1,666+0.990.781
Long 1,667+0.650.714
Context LengthShort 1,667+0.910.763
Medium 1,666+0.920.768
Long 1,667+1.000.782
Response LengthShort 1,667+0.950.771
Medium 1,666+1.180.804
Long 1,667+2.050.893
semantic content. This is geometrically intuitive: a single sentence may be ambiguously positioned, while a
paragraph establishes a clearer location onSd−1.
2. Short questions work better. Effect size decreases fromd= 1.22(short) tod= 0.65(long). Short questions
create tighter semantic anchors. Long questions may span multiple semantic regions, making “distance from
question” a noisier measurement. A question like “What is the capital of France?” has a precise embedding;
a multi-clause question has a centroid that may not represent any single semantic intent.
3. Context length is stable. Effect sizes remain consistent (d≈0.91–1.00) across context lengths. SGI is
robust to context verbosity, likely because the context embedding averages over content in a way that remains
geometrically stable.
These findings shows that SGI is most reliable for long-form responses to focused questions—particularly the setting
where manual verification is most costly.
5.5 Calibration Analysis
Figure 5 shows the calibration analysis. Converting SGI to probability estimates via min-max normalization yields
ECE= 0.10, at the boundary of “well-calibrated.” The reliability diagram shows SGI-derived probabilities track
actual hallucination rates with moderate fidelity.
Figure 5 right plot illustrates the monotonic relationship. Samples in the lowest SGI decile have∼100% hallucination
rate, while those in the highest decile have∼65% rate. The gradient is consistent, proving that SGI can be used as a
probability estimate for risk stratification, not only a binary classifier.
7

APREPRINT- DECEMBER17, 2025
Figure 4: Subgroup robustness analysis. Response length shows the strongest effect: long responses yieldd= 2.05,
nearly double the overall average. Short questions (d= 1.22) outperform long questions (d= 0.65). Context length
has minimal impact.
Figure 5: Calibration analysis. Left: reliability diagram showing SGI probabilities vs. actual hallucination rates (ECE
= 0.10). Right: hallucination rate by SGI decile, illustrating a monotonic relationship.
5.6 Negative Result: TruthfulQA
Table 4 shows the results of our experiments using TruthfulQA dataset. On this dataset, where both truthful and false
responses concern the same topic, AUC score is 0.478—worse than random guessing. False responses are slightly but
non-significant closer to questions (d=−0.14).
This confirms our theoretical prediction that angular distance onSd−1measures topical similarity, not factual ac-
curacy. Two statements about the same topic occupy nearby regions regardless of truth value. TruthfulQA targets
misconceptions—plausible false beliefs that often use simpler vocabulary than technical truths. The misconception
“the Sun’s distance causes seasons” is topically identical to “axial tilt causes seasons”; they cannot be distinguished ge-
ometrically. This negative result is methodologically important because establishes clear boundaries on what angular
geometry can and cannot detect.
Table 4: TruthfulQA results (n= 800). Angular geometry cannot discriminate factual accuracy when both responses
concern the same topic.
Metric Truthful False∆Cohen’sd p-value
θ(r, q)0.782 0.763−0.019−0.140.045
ROC-AUC 0.478 (below chance)
8

APREPRINT- DECEMBER17, 2025
5.7 Signal Decomposition
Table 5: Component analysis: effect sizes forθ(r, q)andθ(r, c)separately. The semantic laziness signal is driven
primarily by hallucinations being closer to questions, not farther from contexts.
Modeld(θ r,q)d(θ r,c)Primary Driver
mpnet+1.50 +0.43θ(r, q)
minilm+1.62 +0.38θ(r, q)
bge+1.48 +0.41θ(r, q)
e5+1.39 +0.45θ(r, q)
gte+1.44 +0.40θ(r, q)
Table 5 decomposes the SGI signal. Across all models, the effect size forθ(r, q)(ranging from+1.39to+1.62)
substantially exceeds that forθ(r, c)(ranging from+0.38to+0.45). This indicates that semantic laziness is driven
primarily by hallucinations beingcloser to questions, not by them beingfarther from contexts.
This asymmetry is theoretically meaningful. When LLMs hallucinate, they are not actively “avoiding” the context
but rather failing to depart from the question’s semantic neighborhood. The generation process defaults to question-
proximate completions when context integration fails.
6 Discussion
6.1 Hallucinations And Uncertainty
Hallucinated responses exhibit a distinctive geometric signature: they cluster angularly near questions rather than
departing toward contexts. This behavior is consistent across five embedding models trained by different organizations,
with mean correlationr= 0.85and ranking agreementρ= 0.87. The effect size increases predictably withθ(q, c)as
the triangle inequality bounds predict. This is the core empirical contribution.
We propose that semantic laziness reflects a default mode of autoregressive generation under uncertainty. When a
model lacks confidence in context integration, it produces completions that remain within the question’s semantic
neighborhood—statistically “safe” territory. This interpretation is plausible but speculative; we have not established
the causal link to internal uncertainty.
If the uncertainty hypothesis is correct, SGI should correlate with internal confidence measures: attention entropy, hid-
den state variance, or logit dispersion. Responses with low SGI should exhibit higher entropy in attention distributions.
We leave this investigation to future work.
6.2 Practical Implications
The experimental results suggest concrete deployment guidelines:
1.Measureθ(q, c)first.Expected discriminative power can be assessed before deployment. Datasets with
smallθ(q, c)will show reduced effect sizes.
2.SGI excels on long responses to short questions.This is precisely where manual verification is most costly,
making SGI particularly valuable for production RAG systems generating detailed answers.
3.Use SGI as probability estimate.With ECE= 0.10, SGI scores can inform risk stratification, not just binary
flagging.
4.Complement with NLI.SGI detects semantic disengagement; NLI detects logical contradiction. The signals
are orthogonal.
6.3 Limitations
We have to acknowledge several limitations in our research. The most important one is the dataset specificity. HaluEval
hallucinations are adversarially generated. Production hallucinations may exhibit different geometric signatures. SGI
captures how responses engage with context, not whether they are correct. The TruthfulQA result shows this limitation.
Besides, SGI assumes the retrieved context is relevant. Poor retrieval undermines the geometric anchor.
Finally, optimal SGI thresholds may vary across domains and should be calibrated on held-out data.
9

APREPRINT- DECEMBER17, 2025
7 Conclusion
We introduced the Semantic Grounding Index, a geometric quantity defined intrinsically on the embedding hyper-
sphereSd−1. Our central finding is that hallucinated responses in RAG systems exhibitsemantic laziness—they
remain angularly proximate to questions rather than departing toward contexts.
The contribution is threefold. First, SGI istheoretically grounded: we derive from the triangle inequality that dis-
criminative power should increase withθ(q, c), and this prediction is confirmed empirically (d= 0.61→1.27across
terciles). Second, SGI isrobust: five embedding models with distinct architectures agree on SGI scores with correla-
tionr= 0.85, indicating that the signal is a property of text rather than embedding geometry. Third, SGI ispractically
characterized: we identify where it excels (long responses, short questions), where it fails (TruthfulQA), and establish
calibration quality (ECE= 0.10).
References
Azaria, A. and Mitchell, T. (2023). The internal state of an LLM knows when it’s lying. InFindings of the Association
for Computational Linguistics: EMNLP 2023, pages 967–976.
Bao, F., Chen, Y ., and Wang, X. (2025). FaithBench: A diverse hallucination benchmark for summarization by modern
LLMs.arXiv preprint arXiv:2501.00942.
Bridson, M.R. and Haefliger, A. (2013).Metric Spaces of Non-Positive Curvature, volume 319 of Grundlehren der
mathematischen Wissenschaften. Springer-Verlag, Berlin.
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., et al. (2020). Language models are few-shot learners. InAdvances in Neural Information Processing
Systems, volume 33, pages 1877–1901.
Catak, F.O., Kuzlu, M., and Guler, O. (2024). Uncertainty quantification in large language models through convex hull
analysis.arXiv preprint arXiv:2406.19712.
Chen, W., Borgeaud, S., and Irving, G. (2024). EigenScore: A simple and effective probe for hallucination detection
using internal representations.arXiv preprint arXiv:2403.17651.
Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y . (2024). Detecting hallucinations in large language models using se-
mantic entropy.Nature, 630(8017):625–630.
Firth, J.R. (1957). A synopsis of linguistic theory, 1930–1955. InStudies in Linguistic Analysis, pages 1–32. Blackwell,
Oxford.
Fisher, R.A. (1953). Dispersion on a sphere.Proceedings of the Royal Society of London. Series A, 217(1130):295–
305.
Gao, T., Yao, X., and Chen, D. (2021). SimCSE: Simple contrastive learning of sentence embeddings. InProceedings
of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910.
Gao, Y ., Zhang, L., and Liu, Y . (2025). Attention-based hallucination detection with trainable kernels.arXiv preprint
arXiv:2506.09886.
Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.-W. (2020). REALM: Retrieval-augmented language model
pre-training. InProceedings of the 37th International Conference on Machine Learning, pages 3929–3938.
HALT-RAG (2025). Task-adaptable hallucination detection with calibrated NLI ensembles.arXiv preprint
arXiv:2509.07475.
Harris, Z.S. (1954). Distributional structure.Word, 10(2–3):146–162.
Honnibal, M., Montani, I., Van Landeghem, S., and Boyd, A. (2020). spaCy: Industrial-strength Natural Language
Processing in Python. DOI: 10.5281/zenodo.1212303.
Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, W., Feng, X., Qin, B., and Liu, T. (2023).
A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions.ACM
Transactions on Information Systems.
Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y ., Ishii, E., Bang, Y .J., Madotto, A., and Fung, P. (2023). Survey of
hallucination in natural language generation.ACM Computing Surveys, 55(12):1–38.
Kovács, Á. and Recski, G. (2025). LettuceDetect: A hallucination detection framework for RAG applications.arXiv
preprint arXiv:2502.17125.
10

APREPRINT- DECEMBER17, 2025
Krishna, K., Roy, A., and Iyyer, M. (2021). Hurdles to progress in long-form question answering. InProceedings
of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics, pages
4940–4957.
Kuhn, L., Gal, Y ., and Farquhar, S. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in
natural language generation. InThe Eleventh International Conference on Learning Representations.
Laban, P., Schnabel, T., Bennett, P.N., and Hearst, M.A. (2022). SummaC: Re-visiting NLI-based models for incon-
sistency detection in summarization.Transactions of the Association for Computational Linguistics, 10:163–177.
Ledoux, M. (2001).The Concentration of Measure Phenomenon, volume 89 of Mathematical Surveys and Mono-
graphs. American Mathematical Society, Providence, RI.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V ., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel,
T., Riedel, S., and Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. InAd-
vances in Neural Information Processing Systems, volume 33, pages 9459–9474.
Li, J., Cheng, X., Zhao, W.X., Nie, J.-Y ., and Wen, J.-R. (2023). HaluEval: A large-scale hallucination evaluation
benchmark for large language models. InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing, pages 6449–6464.
Li, Z., Zhang, X., Zhang, Y ., Long, D., Xie, P., and Zhang, M. (2023). Towards general text embeddings with multi-
stage contrastive learning.arXiv preprint arXiv:2308.03281.
Li, X., Wang, Y ., and Chen, Z. (2025). Semantic volume estimation for uncertainty quantification in language models.
arXiv preprint arXiv:2501.08765.
Lin, S., Hilton, J., and Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. InProceedings
of the 60th Annual Meeting of the Association for Computational Linguistics, pages 3214–3252.
Mardia, K.V . and Jupp, P.E. (2000).Directional Statistics. Wiley Series in Probability and Statistics. John Wiley &
Sons, Chichester.
Meng, Y ., Huang, J., Zhang, G., and Han, J. (2019). Spherical text embedding. InAdvances in Neural Information
Processing Systems, volume 32, pages 8208–8217.
Niu, L., Jia, F., Wu, S., and Chen, Y . (2024). RAGTruth: A hallucination corpus for developing trustworthy retrieval-
augmented generation.arXiv preprint arXiv:2401.00396.
Pestov, V . (2000). On the geometry of similarity search: Dimensionality curse and concentration of measure.Informa-
tion Processing Letters, 73(1–2):47–51.
Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. InPro-
ceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, pages 3982–3992.
Shuster, K., Poff, S., Chen, M., Kiela, D., and Weston, J. (2021). Retrieval augmentation reduces hallucination in
conversation. InFindings of the Association for Computational Linguistics: EMNLP 2021, pages 3784–3803.
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E.,
Azhar, F., et al. (2023). LLaMA: Open and efficient foundation language models.arXiv preprint arXiv:2302.13971.
Wang, T. and Isola, P. (2020). Understanding contrastive representation learning through alignment and uniformity on
the hypersphere. InProceedings of the 37th International Conference on Machine Learning, pages 9929–9939.
Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and Zhou, M. (2022). MiniLM: Deep self-attention distillation
for task-agnostic compression of pre-trained transformers. InAdvances in Neural Information Processing Systems,
volume 33, pages 5776–5788.
Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., and Wei, F. (2024). Text embeddings by
weakly-supervised contrastive pre-training.arXiv preprint arXiv:2212.03533.
Xiao, S., Liu, Z., Zhang, P., and Muennighoff, N. (2023). C-Pack: Packaged resources to advance general Chinese
embedding.arXiv preprint arXiv:2309.07597.
You, K. (2025). Semantics at an angle: When cosine similarity works (and when it doesn’t).arXiv preprint
arXiv:2504.16318.
11