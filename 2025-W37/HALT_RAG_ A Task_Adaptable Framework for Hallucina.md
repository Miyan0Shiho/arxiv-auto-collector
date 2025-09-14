# HALT-RAG: A Task-Adaptable Framework for Hallucination Detection with Calibrated NLI Ensembles and Abstention

**Authors**: Saumya Goswami, Siddharth Kurra

**Published**: 2025-09-09 07:58:46

**PDF URL**: [http://arxiv.org/pdf/2509.07475v1](http://arxiv.org/pdf/2509.07475v1)

## Abstract
Detecting content that contradicts or is unsupported by a given source text
is a critical challenge for the safe deployment of generative language models.
We introduce HALT-RAG, a post-hoc verification system designed to identify
hallucinations in the outputs of Retrieval-Augmented Generation (RAG)
pipelines. Our flexible and task-adaptable framework uses a universal feature
set derived from an ensemble of two frozen, off-the-shelf Natural Language
Inference (NLI) models and lightweight lexical signals. These features are used
to train a simple, calibrated, and task-adapted meta-classifier. Using a
rigorous 5-fold out-of-fold (OOF) training protocol to prevent data leakage and
produce unbiased estimates, we evaluate our system on the HaluEval benchmark.
By pairing our universal feature set with a lightweight, task-adapted
classifier and a precision-constrained decision policy, HALT-RAG achieves
strong OOF F1-scores of 0.7756, 0.9786, and 0.7391 on the summarization, QA,
and dialogue tasks, respectively. The system's well-calibrated probabilities
enable a practical abstention mechanism, providing a reliable tool for
balancing model performance with safety requirements.

## Full Text


<!-- PDF content starts -->

HALT-RAG: A Task-Adaptable Framework for Hallucination
Detection
with Calibrated NLI Ensembles and Abstention
Saumya Goswami Siddharth Kurra
Abstract
Detecting content that contradicts or is unsupported by a
given source text is a critical challenge for the safe deploy-
ment of generative language models. We introduceHALT-
RAG, a post-hoc verification system designed to identify
hallucinations in the outputs of Retrieval-Augmented Gen-
eration (RAG) pipelines. Our flexible and task-adaptable
framework uses a universal feature set derived from an
ensemble of two frozen, off-the-shelf Natural Language
Inference (NLI) models and lightweight lexical signals.
These features are used to train a simple, calibrated, and
task-adapted meta-classifier. Using a rigorous 5-fold out-
of-fold (OOF) training protocol to prevent data leakage
and produce unbiased estimates, we evaluate our system
on the HALUEVALbenchmark. By pairing our univer-
sal feature set with a lightweight, task-adapted classifier
and a precision-constrained decision policy, HALT-RAG
achieves strong OOF F1-scores of 0.7756, 0.9786, and
0.7391 on the summarization, QA, and dialogue tasks,
respectively. The system’s well-calibrated probabilities
enable a practical abstention mechanism, providing a re-
liable tool for balancing model performance with safety
requirements.
1 Introduction
Modern generative NLP models often producehallucina-
tions: content that is not supported by the input source or
external knowledge [ 2,3]. This problem is widespread;
studies report that standard models for abstractive sum-
marization on the XSum dataset produce factually consis-
tent summaries only 20–30% of the time [ 9]. Detecting
these inconsistencies is therefore critical for deploying
safe and reliable NLP systems, especially for **Retrieval-
Augmented Generation (RAG)** pipelines, where output
quality depends entirely on the retrieved context. HALT-
RAG is designed as a post-hoc verifier for such systems,
operating on the source text and generated output without
performing retrieval itself.
However, existing metrics for factual consistency haveclear limitations. Many aretask-specific, such as NLI-
based detectors like FactCC [ 6] and SummaC [ 7] for sum-
marization. Othersrely on heuristic or synthetically
generated training data[ 6,8], which can limit their ap-
plicability. Furthermore, many produceuncalibrated con-
fidence scoresthat are hard to interpret [ 5], making them
a poor fit for applications that need a ”reject option” [1].
In this work, we present a principled framework for
building and evaluating a reliable hallucination detector.
Our system,HALT-RAG, is built on a universal feature
set derived from adual-NLI predictor ensembleand
simple lexical signals. This combined signal is fed into
acalibrated meta-classifier. By training with anout-of-
fold (OOF) strategyand applyingpost-hoc calibration
[10,13], we obtain well-calibrated confidence estimates
that enable precise, policy-driven decisions.
Our main contributions are:
1.We demonstrate that an ensemble of frozen, off-the-
shelf NLI models, combined with simple lexical statis-
tics, forms a powerful and efficient universal feature
set. This serves as a strong input to a lightweight,
task-adapted classifier, avoiding the need to fine-tune
large language models.
2.We introduce a principled evaluation protocol using
out-of-fold prediction and post-hoc calibration to pro-
duce reliable, unbiased performance estimates suit-
able for deploying safety-critical systems.
3.We show that our framework, pairing a universal fea-
ture set with a task-adapted classifier and a precision-
constrained thresholding policy (Precision ≥0.70 ),
achieves strong performance on the diverse tasks
within the HALUEVALbenchmark.
2 Related Work
2.1 Hallucination in NLP Tasks
Hallucination is a well-documented problem across several
NLP domains.
1arXiv:2509.07475v1  [cs.CL]  9 Sep 2025

0.0 0.2 0.4 0.6 0.8 1.0
Recall020406080Precision
F1=0.6F1=0.7F1=0.8F1=0.9Summarization  PR (N=20000)
Bootstrap 95% band
HALT-RAG (AUPR=0.844)
t*=0.377  P=0.712, R=0.851
Precision target = 0.70(a) Precision-Recall Curve
0.0 0.2 0.4 0.6 0.8 1.0
False Positive Rate0.00.20.40.60.81.0True Positive RateSummarization  ROC (N=20000)
Bootstrap 95% band
HALT-RAG (AUROC=0.847)
t*=0.377  TPR=0.851, FPR=0.344
 (b) ROC Curve
0.0 0.2 0.4 0.6 0.8 1.0
Predicted probability of hallucination0.00.20.40.60.81.0Empirical fraction hallucinated
Summarization  Reliability (N=20000)
Perfect calibration
Observed (ECE=0.011)
(c) Calibration Curve (ECE = 0.011)
0.2 0.4 0.6 0.8 1.0
Coverage (kept fraction)0.700.750.800.850.900.95ScoreSummarization  Risk Coverage (abstain by uncertainty around t*)
F1 vs coverage
Precision vs coverage
90% cov  F1=0.800, P=0.737
 (d) Risk-Coverage Trade-off
Figure 1: Performance of HALT-RAG on theSummarizationtask. The markers on the PR/ROC curves show the
operating point at the chosen threshold ( t∗= 0.377 ). The strong calibration (c) and graceful trade-off between
precision/F1 and coverage (d) highlight the model’s reliability.
2.1.1 Abstractive Summarization
In abstractive summarization, models are highly prone to
generating unfaithful content [ 2,9]. Maynez et al. [ 9]
reported that on the XSum dataset, over 70% of single-
sentence summaries contain hallucinations. Early work to
detect these inconsistencies involved NLI-based detectors
like FactCC [ 6] and SummaC [ 7], which frame the problem
as an entailment task. SummaC noted that prior models
often failed because of a ”mismatch in input granularity”
between sentence-level NLI datasets and document-level
consistency checking [ 7]. At the same time, QA-based met-
rics like QAGS [ 11] and FEQA [ 2] have shown stronger
correlation with human judgments than simple lexical over-lap.
2.1.2 Knowledge-Grounded Dialogue
In dialogue systems, a common failure is the production of
”unsupported utterances,” or hallucinations [ 3]. Dziri et al.
found that many conversational datasets contain halluci-
nations and introducedFaithDial, a benchmark designed
to mitigate this issue [ 3]. FaithDial also provides training
signals for hallucination critics, like FAITHCRITIC, to
help discriminate utterance faithfulness. Honovich et al.
[4] proposedQ2, a QA-inspired metric that compares an-
swer spans using NLI instead of token matching, making
it more robust to lexical variation.
2

0.0 0.2 0.4 0.6 0.8 1.0
Recall020406080Precision
F1=0.6F1=0.7F1=0.8F1=0.9Qa  PR (N=20000)
Bootstrap 95% band
HALT-RAG (AUPR=0.996)
t*=0.395  P=0.984, R=0.974
Precision target = 0.70(a) Precision-Recall Curve
0.0 0.2 0.4 0.6 0.8 1.0
False Positive Rate0.00.20.40.60.81.0True Positive RateQa  ROC (N=20000)
Bootstrap 95% band
HALT-RAG (AUROC=0.995)
t*=0.395  TPR=0.974, FPR=0.016
 (b) ROC Curve
0.0 0.2 0.4 0.6 0.8 1.0
Predicted probability of hallucination0.00.20.40.60.81.0Empirical fraction hallucinated
Qa  Reliability (N=20000)
Perfect calibration
Observed (ECE=0.005)
(c) Calibration Curve (ECE = 0.005)
0.2 0.4 0.6 0.8 1.0
Coverage (kept fraction)0.9800.9850.9900.9951.000ScoreQa  Risk Coverage (abstain by uncertainty around t*)
F1 vs coverage
Precision vs coverage
90% cov  F1=0.993, P=0.995
 (d) Risk-Coverage Trade-off
Figure 2: Performance of HALT-RAG on theQuestion Answeringtask. The detector achieves near-perfect discrimina-
tion, with its chosen threshold ( t∗= 0.395 ) operating at very high precision and recall. The extremely low ECE of
0.005 indicates excellent calibration.
2.1.3 Open-Domain Question Answering
In QA, models need to know ”when to abstain from an-
swering” to avoid providing incorrect information, espe-
cially under domain shift [ 5]. Kamath et al. [ 5] showed
that policies based on simple softmax probabilities per-
form poorly because models are oftenoverconfident on
out-of-domain inputs, highlighting the need for robust
calibration.
2.1.4 Hallucination Evaluation Benchmarks
HALUEVAL[ 8] is a key recent benchmark that includes a
”large collection of generated and human-annotated hallu-
cinated samples” for summarization, QA, and dialogue. Itsmulti-task, human-annotated nature makes it an ideal plat-
form for evaluating general-purpose detectors like HALT-
RAG.
2.2 Contemporary Approaches to Hallucina-
tion in RAG
Recent work (late 2023-2024) has explored methods more
deeply integrated into the RAG process itself. Some ap-
proaches perform real-time hallucination detection to dy-
namically alter the generation process, while others ana-
lyze the internal states of the generator model for consis-
tency. Another line of research focuses on making RAG
systems robust to the noisy or irrelevant context often re-
3

0.0 0.2 0.4 0.6 0.8 1.0
Recall020406080Precision
F1=0.6F1=0.7F1=0.8F1=0.9Dialogue  PR (N=20000)
Bootstrap 95% band
HALT-RAG (AUPR=0.829)
t*=0.421  P=0.706, R=0.776
Precision target = 0.70(a) Precision-Recall Curve
0.0 0.2 0.4 0.6 0.8 1.0
False Positive Rate0.00.20.40.60.81.0True Positive RateDialogue  ROC (N=20000)
Bootstrap 95% band
HALT-RAG (AUROC=0.819)
t*=0.421  TPR=0.776, FPR=0.323
 (b) ROC Curve
0.0 0.2 0.4 0.6 0.8 1.0
Predicted probability of hallucination0.00.20.40.60.81.0Empirical fraction hallucinated
Dialogue  Reliability (N=20000)
Perfect calibration
Observed (ECE=0.013)
(c) Calibration Curve (ECE = 0.013)
0.2 0.4 0.6 0.8 1.0
Coverage (kept fraction)0.700.750.800.850.900.95ScoreDialogue  Risk Coverage (abstain by uncertainty around t*)
F1 vs coverage
Precision vs coverage
90% cov  F1=0.763, P=0.730
 (d) Risk-Coverage Trade-off
Figure 3: Performance of HALT-RAG on theDialoguetask. This task is the most challenging due to conversational
nuances, but the model still maintains strong, well-calibrated performance at its chosen threshold (t∗= 0.421).
turned by the retriever module. In contrast to these more in-
tegrated approaches, HALT-RAG is designed as a comple-
mentary, retrieval-agnostic,post-hoc verifier. Its strength
lies in its ability to serve as a robust final check in any RAG
pipeline, regardless of the retrieval or generation strategy.
2.3 Factual Verification and Selective An-
swering
The ”reject option” allows a model to abstain when its
confidence is low [ 5]. This idea, formalized by Chow
(1970), converts potential misclassifications into rejections
[1]. For this to work well, models must providewell-
calibrated confidence estimates. Post-hoc calibration
methods [ 12,13], such as Platt scaling for SVMs [ 10],are crucial for turning raw model outputs into reliable
probabilities that enable dependable abstention decisions.
3 Methodology
The HALT-RAG pipeline has three stages: feature extrac-
tion, meta-classification, and calibration.
3.1 Model Architecture
NLI and Lexical Features.We segment the source
and generated text into premise-hypothesis pairs us-
ing anon-overlapping windowof 320 tokens with
a stride of 320. Each pair is scored by two
4

frozen NLI models: roberta-large-mnli and
microsoft/deberta-v3-large . We chose this en-
semble to leverage their architectural diversity. RoBERTa
is a highly optimized BERT-style model, while DeBERTa’s
disentangled attention mechanism separates content and
position embeddings. We hypothesized that combining
these distinct approaches would yield a more robust se-
mantic signal, which was validated by our ablation study
(Table 2). These windowed NLI probabilities are then
summarized usingmax and mean poolingand concate-
nated with lightweight lexical features, including sequence
lengths, length ratios,ROUGE-L overlap, andJaccard
similarity.
Meta-Classification.This combined feature vector is
fed into a simple meta-classifier. While our feature set
is designed to be task-agnostic, we found that optimal
performance required adapting the classifier to the task.
We use aLogistic Regressionmodel with balanced class
weights for the Summarization and Dialogue tasks. For the
Question Answering (QA) task, which exhibited a more
linearly separable feature space, aLinear Support Vector
Classifier (LinearSVC)achieved superior performance.
A LinearSVC optimizes a hinge loss to find a max-margin
hyperplane, effective for clearly separated data, while Lo-
gistic Regression optimizes log-loss for probabilistic out-
puts. This finding is supported by a t-SNE visualization
of the feature space in the Appendix. This task-adapted
approach highlights that the nature of hallucination differs
across tasks, and the choice of a simple classifier can be
tailored to these differences without altering the underlying
feature representation.
Calibration Protocol.We use a5-fold out-of-fold
(OOF) training protocolto generate unbiased predictions
for the entire training set. A final calibration model is fit on
these OOF predictions. Our choice of method aligns with
established best practices: for the QA task’s LinearSVC,
we usePlatt scaling[ 10], a parametric method well-suited
for SVMs. For the other tasks, we useisotonic regression,
a more powerful non-parametric method that benefits from
larger datasets.
3.2 Thresholding and Abstention Strategy
Threshold Optimization.The threshold optimization
objective is to find t∗= arg max tF1(t)subject to
Precision(t)≥π 0. The final thresholds are tSumm =
0.377,t QA= 0.395, andt Dial= 0.421.
Selective Prediction (Abstention).Because our prob-
abilities are calibrated, we can use a robust abstention
mechanism. A user can define acoverage target (e.g.,90%)and apply a stricter threshold to reject predictions
that fall into an uncertainty band, trading a small amount
of coverage for a significant gain in precision.
4 Experiments and Analysis
4.1 Experimental Setup
We evaluate HALT-RAG on theHALUEVALbenchmark
[8] for Summarization, QA, and Dialogue. All reported
metrics are computed on theout-of-fold predictionsfrom
our 5-fold cross-validation setup, ensuring evaluation is
performed on data unseen by each fold-specific model.
4.2 Main Results
As shown in Table 1, HALT-RAG performs well across
the board. It is exceptionally strong on QA, with an F1-
score of0.9786, and also robust on the more challenging
Summarization and Dialogue tasks. The calibration plots
(Figures 1 to 3) show alow Expected Calibration Error
(ECE)across all tasks (0.011 for Summarization, 0.005 for
QA, and 0.013 for Dialogue), which confirms the reliability
of its confidence scores.
Table 1: Main performance metrics on the HaluEval
benchmark (Out-of-Fold), optimized for F1with Preci-
sion≥0.70.
Task Threshold Precision RecallF 1-Score Accuracy
Summarization0.377 0.7122 0.8514 0.7756 0.7537
QA0.395 0.9838 0.9735 0.9786 0.9788
Dialogue0.421 0.7059 0.7756 0.7391 0.7262
4.3 Ablation Studies
We ran ablation experiments on the Summarization devel-
opment set (Table 2). The results show how much each
component contributes:
•Removing thecontradiction or entailment signals
from the NLI models drops the F1-score by2.1 and
4.5 points, respectively.
•Removing thelightweight lexical featureshurts per-
formance by1.3 points.
•Using just asingle NLI model(DeBERTa) reduces
theF1-score by1.8 pointscompared to the full en-
semble.
These results confirm that each component of our archi-
tecture is essential for its performance. The fact that re-
moving the entailment signal has a greater impact suggests
that positively identifying supporting evidence is a more
5

critical signal for factuality than merely detecting contra-
dictions. However, this finding may also be an artifact
of the HALUEVALdataset itself, where it may be more
common for models to generate plausible but unsupported
statements (lacking entailment) than direct logical contra-
dictions.
Table 2: Ablation experiments on the HaluEval Summa-
rization development set.
Model Variant Precision RecallF 1-Score
Full HALT-RAG model0.705 0.844 0.768
– without Contradiction Signal0.673 0.839 0.747
– without Entailment Signal0.651 0.810 0.723
– without Lexical Features0.694 0.831 0.755
Single NLI (DeBERTa only)0.691 0.820 0.750
4.4 Impact of Abstention
Our model’s calibration enables effective selective predic-
tion. As shown in Table 3, byabstaining on the 10% of
examples with the lowest confidence, we can significantly
increase precision with only a minor dip in the F1-score.
For Summarization, precision jumps by8.6 points(from
0.712 to 0.798); for Dialogue, by7.7 points(from 0.706
to 0.783); and for QA, it improves from 0.984 to0.998.
This shows the practical value of the reject option for high-
stakes applications.
5 Discussion and Future Work
Failure Analysis.The model’s near-perfect performance
on QA is likely due to the task’s nature: QA pairs are often
short, self-contained, and have low semantic ambiguity. In
contrast, the Dialogue task is the most challenging ( F1of
0.7391). This performance gap can be directly attributed
to the limitations of our fixed-windowing approach. While
a 320-token window is sufficient for self-contained QA
pairs, it is inherently incapable of capturing the long-range
context, coreferences, and pragmatic effects required to
verify factuality in multi-turn dialogue.
Limitations.Our study has several limitations that pro-
vide clear avenues for future work. First, while our feature
set is shared across tasks, optimal performance required
task-adapted meta-classifiers and decision thresholds, indi-
cating the final system is task-adapted rather than univer-
sally generalizable. Second, our feature extraction relies
on non-overlapping 320-token windows, which cannot de-
tect inconsistencies that span window boundaries. Third,
our validation is confined to the HALUEVALbenchmark.
More critically, we identify two further limitations:•Computational Cost and Latency:We do not ana-
lyze the computational overhead of HALT-RAG. Run-
ning two large transformer models across multiple
windows can introduce significant latency, a trade-off
for accuracy that could be a consideration for real-
time applications.
•Robustness to Noisy Retrieval:Our system was
evaluated on HALUEVAL’s clean, relevant source doc-
uments. Its performance against noisy or irrelevant
retrieved context—a primary challenge in real-world
RAG—is an important and unevaluated area for future
work.
Ethical Considerations.For sensitive applications, we
recommend usingconservative decision thresholdsand
keeping ahuman-in-the-loopfor low-confidence predic-
tions.
Future Work.Future research could incorporate more
sophisticated semantic features, such asentity linking and
relation extraction, to better ground the detector in real-
world facts. A critical next step for this line of work is to
integrate and evaluate detectors like HALT-RAG within
a live RAG pipeline to measure its resilience tonoisy or
irrelevant retrieved context, which is a key challenge in
production systems.
6 Conclusion
We presented HALT-RAG, atask-adaptable framework
for hallucination detectionthat combines a dual-NLI en-
semble, lexical features, and a calibrated meta-classifier.
Our approach delivers high-precision, reliable performance
on the HALUEVALbenchmark. Through a principled
thresholding strategy and an effective abstention mech-
anism, HALT-RAG provides a practical tool for improving
the factuality and safety of modern generative NLP sys-
tems. All artifacts are released to ensure reproducibility.
References
[1]C. K. Chow. 1970. On optimum recognition error and
reject tradeoff.IEEE Transactions on Information
Theory, 16(1):41–46.
[2]Esin Durmus, He He, and Mona Diab. 2020. FEQA:
A question answering evaluation framework for faith-
fulness assessment in abstractive summarization. In
Proceedings of the 58th Annual Meeting of the Asso-
ciation for Computational Linguistics, pages 5055–
5070, Online. Association for Computational Lin-
guistics.
6

Table 3: Performance with and without abstention. At∼90% coverage, precision is substantially improved.
Task Setting Coverage (%) PrecisionF 1-Score
SummarizationStandard100.0 0.7122 0.7756
Selective89.40.79800.7820
QAStandard100.0 0.9838 0.9786
Selective90.60.99800.9800
DialogueStandard100.0 0.7059 0.7391
Selective90.20.78300.7240
[3]Nouha Dziri, Ehsan Kamalloo, Sivan Milton, Os-
mar Zaiane, Mo Yu, Edoardo M. Ponti, and Siva
Reddy. 2022. FaithDial: A Faithful Benchmark for
Information-Seeking Dialogue.Transactions of the
Association for Computational Linguistics, 10:1473–
1490.
[4]Or Honovich, Leshem Choshen, Roee Aharoni, Elad
Neeman, Idan Szpektor, and Omri Abend. 2021.
Q2: Evaluating factual consistency in knowledge-
grounded dialogues via question generation and ques-
tion answering. InProceedings of the 2021 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, pages 7856–7870, Online and Punta Cana,
Dominican Republic. Association for Computational
Linguistics.
[5]Amita Kamath, Robin Jia, and Percy Liang. 2020.
Selective question answering under domain shift. In
Proceedings of the 58th Annual Meeting of the Asso-
ciation for Computational Linguistics, pages 5674–
5688, Online. Association for Computational Lin-
guistics.
[6]Wojciech Kry ´sci´nski, Bryan McCann, Caiming
Xiong, and Richard Socher. 2020. Evaluating the
factual consistency of abstractive text summarization.
InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP),
pages 9334–9346, Online. Association for Computa-
tional Linguistics.
[7]Philippe Laban, Tobias Schnabel, Paul N. Bennett,
and Marti A. Hearst. 2022. SummaC: Re-Visiting
NLI-based Models for Inconsistency Detection in
Summarization.Transactions of the Association for
Computational Linguistics, 10:163–177.
[8]Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie,
and Ji-Rong Wen. 2023. HaluEval: A large-scale hal-
lucination evaluation benchmark for large language
models. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,pages 6494–6512, Singapore. Association for Com-
putational Linguistics.
[9]Joshua Maynez, Shashi Narayan, Bernd Bohnet, and
Ryan McDonald. 2020. On faithfulness and factu-
ality in abstractive summarization. InProceedings
of the 58th Annual Meeting of the Association for
Computational Linguistics, pages 1906–1919, On-
line. Association for Computational Linguistics.
[10] John C. Platt. 1999. Probabilistic outputs for sup-
port vector machines and comparisons to regularized
likelihood methods. Technical report, Microsoft Re-
search. Published in Advances in Large Margin Clas-
sifiers.
[11] Alex Wang, Kyunghyun Cho, and Mike Lewis. 2020.
Asking and answering questions to evaluate the fac-
tual consistency of summaries. InProceedings of the
58th Annual Meeting of the Association for Compu-
tational Linguistics, pages 5008–5020, Online. Asso-
ciation for Computational Linguistics.
[12] Bianca Zadrozny and Charles Elkan. 2001. Obtaining
calibrated probability estimates from decision trees
and naive bayesian classifiers. InProceedings of
the Eighteenth International Conference on Machine
Learning, pages 609–616.
[13] Bianca Zadrozny and Charles Elkan. 2002. Trans-
forming classifier scores into accurate multiclass
probability estimates. InProceedings of the Eighth
ACM SIGKDD International Conference on Knowl-
edge Discovery and Data Mining, pages 694–699.
A Reproducibility
All code, data processing scripts, and evaluation logic re-
quired to reproduce the results in this paper are available
in the public GitHub repository: https://github.
com/sgoswami06/halt-rag/ . The following sec-
tions provide detailed instructions and specifications.
7

A.1 Execution Commands
Assuming the repository has been cloned and all depen-
dencies from ‘requirements.txt‘ have been installed in a
suitable Python environment, the main results reported in
Table 1 can be reproduced by executing the top-level Make-
file targets. Each command runs the full pipeline: feature
extraction, OOF training, calibration, and evaluation for
the specified task.
# To reproduce Summarization results
make eval-summarization
# To reproduce Question Answering results
make eval-qa
# To reproduce Dialogue results
make eval-dialogue
A.2 Hyperparameters
The meta-classifiers were implemented using
scikit-learn . Key hyperparameters for each
task are detailed in Table 4. The regularization strength
was left at the library’s default value of C= 1.0 for all
models.
Table 4: Meta-classifier hyperparameters for each task.
Hyperparameter Summarization QA Dialogue
Model Type Logistic Regression LinearSVC Logistic Regression
Regularization (C) 1.0 1.0 1.0
Class Weight ‘balanced‘ ‘balanced‘ ‘balanced‘
Calibration Isotonic Regression Platt Scaling Isotonic Regression
A.3 Artifact Inventory
The execution of each make command generates a set of ar-
tifacts in the evaluation results/<task name>/
directory. The key output files are:
•oofcalibrated pred.jsonl : Contains the
full set of out-of-fold predictions. Each line is a JSON
object with keys such as id,rawscore (from the
base model), calibrated prob (the final proba-
bility), andlabel.
•oofmeta.json : A summary file containing
the final aggregate metrics. The values for Preci-
sion, Recall, F1-Score, and the optimized Thresh-
old reported in Table 1 are located in this file un-
der keys like precision atprec ge0.70 and
f1atprec ge0.70.
•plots/ : A directory containing all figures shown in
the paper, including Precision-Recall, ROC, Calibra-
tion, and Risk-Coverage curves.A.4 Sanity Checks
•Data Purity:The out of fold protocol is implemented
using sklearn.model selection.KFold .
Critically the entire feature generation process is
completed *before* the cross-validation loop begins.
This design choice is the primary safeguard against
data leakage, as it ensures that each fold’s model is
trained on a completely disjoint set of indices from
its validation set, with no possibility of information
from the validation data influencing feature creation.
•Class Balance:The HaluEval dataset provides rea-
sonably balanced classes for each task, making met-
rics like Accuracy andF 1-score appropriate for eval-
uation.
•Calibration Monotonicity:The generated calibra-
tion plots (Figures 1 to 3) confirm that the isotonic
and Platt scaling steps produce monotonically increas-
ing functions, mapping raw scores to well-behaved
probabilities as expected.
8