# Chronocept: Instilling a Sense of Time in Machines

**Authors**: Krish Goel, Sanskar Pandey, KS Mahadevan, Harsh Kumar, Vishesh Khadaria

**Published**: 2025-05-12 15:07:32

**PDF URL**: [http://arxiv.org/pdf/2505.07637v1](http://arxiv.org/pdf/2505.07637v1)

## Abstract
Human cognition is deeply intertwined with a sense of time, known as
Chronoception. This sense allows us to judge how long facts remain valid and
when knowledge becomes outdated. Despite progress in vision, language, and
motor control, AI still struggles to reason about temporal validity. We
introduce Chronocept, the first benchmark to model temporal validity as a
continuous probability distribution over time. Using skew-normal curves fitted
along semantically decomposed temporal axes, Chronocept captures nuanced
patterns of emergence, decay, and peak relevance. It includes two datasets:
Benchmark I (atomic facts) and Benchmark II (multi-sentence passages).
Annotations show strong inter-annotator agreement (84% and 89%). Our baselines
predict curve parameters - location, scale, and skewness - enabling
interpretable, generalizable learning and outperforming classification-based
approaches. Chronocept fills a foundational gap in AI's temporal reasoning,
supporting applications in knowledge grounding, fact-checking,
retrieval-augmented generation (RAG), and proactive agents. Code and data are
publicly available.

## Full Text


<!-- PDF content starts -->

arXiv:2505.07637v1  [cs.CL]  12 May 2025Chronocept: Instilling a Sense of Time in Machines
Krish Goel*, Sanskar Pandey*, KS Mahadevan, Harsh Kumar, Vishesh Khadaria
krish@projectendgame.tech, pandeysanskar854@gmail.com, mahadevanks26@gmail.com,
kumarharsh3014@gmail.com, khadariavishesh@gmail.com
Abstract
Human cognition is deeply intertwined with a
sense of time, known as Chronoception . This
sense allows us to judge how long facts re-
main valid and when knowledge becomes out-
dated. Despite progress in vision, language,
and motor control, AI still struggles to reason
about temporal validity. We introduce Chrono-
cept, the first benchmark to model temporal
validity as a continuous probability distribution
over time. Using skew-normal curves fitted
along semantically decomposed temporal axes,
Chronocept captures nuanced patterns of emer-
gence, decay, and peak relevance. It includes
two datasets: Benchmark I (atomic facts) and
Benchmark II (multi-sentence passages). Anno-
tations show strong inter-annotator agreement
(84% and 89%). Our baselines predict curve
parameters - location, scale, and skewness - en-
abling interpretable, generalizable learning and
outperforming classification-based approaches.
Chronocept fills a foundational gap in AI’s
temporal reasoning, supporting applications in
knowledge grounding, fact-checking, retrieval-
augmented generation (RAG), and proactive
agents. Code and data are publicly available.
1 Introduction
Humans effortlessly track how information changes
in relevance over time. We instinctively know when
facts emerge, become useful, or fade into obsoles-
cence - a cognitive ability known as Chronoception
(Fontes et al., 2016; Zhou et al., 2019). This higher-
order perception of time plays a crucial role in how
we evaluate the persistence and usefulness of infor-
mation in real-world contexts. Despite excelling
in pattern recognition (He et al., 2016), language
generation (Brown et al., 2020), and motor control
(Levine et al., 2016), modern AI systems remain
largely insensitive to the temporal validity of the
information they process.
*Equal contribution.Prior work has advanced temporal understanding
via event ordering (Allen, 1983; Ning et al., 2020;
Wen and Ji, 2021), timestamp prediction (Kan-
habua and Nørvåg, 2008; Kumar et al., 2012; Das
et al., 2017), and temporal commonsense reasoning
(Zhou et al., 2019). However, these approaches of-
ten reduce time to static labels or binary transitions.
Even recent efforts in temporal validity change pre-
diction (Wenzel and Jatowt, 2024) model shifts as
discrete class changes, neglecting the gradual and
asymmetric nature of temporal decay.
We introduce Chronocept, a benchmark that
models temporal validity as a continuous probabil-
ity distribution over time. Using a skewed-normal
distribution over logarithmic time, parameterized
by location ( ξ), scale ( ω), and skewness ( α) (Az-
zalini, 1986; Schmidt et al., 2017), Chronocept
captures subtle temporal patterns such as delayed
peaks and asymmetric decay.
To support structured supervision, we decom-
pose each sample along semantic temporal axes.
We release two benchmarks: Benchmark I features
atomic factual statements, and Benchmark II con-
tains multi-sentence passages with temporally in-
terdependent elements. High inter-annotator agree-
ment across segmentation, axis labeling, and curve
parameters validates annotation quality.
We benchmark a diverse set of models, includ-
ing linear regression, SVMs, XGBoost, FFNNs, Bi-
LSTMs, and BERT (Devlin et al., 2019). FFNNs
perform best on the simpler Benchmark I, while Bi-
LSTMs excel on the more complex Benchmark II.
Surprisingly, fine-tuned BERTs do not outperform
simpler architectures. To assess the role of tempo-
ral structure, we conduct ablations that remove or
shuffle temporal axes during training - both lead to
marked performance drops.
Chronocept enables several downstream applica-
tions. In Retrieval-Augmented Generation (RAG),
temporal curves guide time-sensitive retrieval; in
fact-checking, they help flag decaying or stale facts.
1

Most importantly, Chronocept lays the foundation
for proactive AI systems that reason not just about
what to do, but when to do it (Miksik et al., 2020).
All resources - dataset, and baseline implemen-
tations - are publicly available to support future
research in machine time-awareness.
2 Related Work
2.1 Temporal Validity Prediction
In the earliest attempt to formalize the temporal va-
lidity of information, Takemura and Tajima (2012)
proposed the concept of “content viability” by clas-
sifying tweets into “read now,” “read later,” and
“expired” categories, to prioritize timeliness in in-
formation consumption. However, their approach
assumed a rigid, monotonic decay of relevance, fail-
ing to model scenarios where validity peaks later
rather than at publication. This restricted its appli-
cability beyond real-time contexts such as Twitter
streams.
Almquist and Jatowt (2019) extended this work
by defining a “validity period” and effectively
proposing a “content expiry date” for sentences,
using linguistic and statistical features. However,
their reliance on static time classes (e.g., hours,
days, weeks) sacrificed granularity, and their ap-
proach required explicit feature engineering rather
than leveraging more advanced, data-driven meth-
ods (Das et al., 2017).
Traditional approaches (Almquist and Jatowt,
2019; Lynden et al., 2023; Hosokawa et al., 2023)
mostly treat validity as binary, where information
is either valid or invalid at a given time, this can be
formulated as:
validityi(t) =(
True if information iis valid at t,
False otherwise(1)
where irepresents the information under consid-
eration and tdenotes the time at which its validity
is evaluated. However, this model overlooks grad-
ual decay, recurrence, and asymmetric relevance
patterns.
More recently, Wenzel and Jatowt (2024) in-
troduced Temporal Validity Change Prediction
(TVCP), which models how context alters a state-
ment’s validity window. However, it does not quan-
tify validity as a continuous probability distribution
over time.
Chronocept advances this field by defining tem-
poral validity as a continuous probability distribu-tion, allowing a more precise and flexible represen-
tation of how information relevance evolves.
2.2 Temporal Reasoning and Commonsense
Temporal reasoning has largely focused on event or-
dering (Allen, 1983; Wen and Ji, 2021; Ning et al.,
2020), predicting temporal context (Kanhabua and
Nørvåg, 2008; Kumar et al., 2012; Das et al., 2017;
Luu et al., 2021; Jatowt et al., 2013), and com-
monsense knowledge (Zhou et al., 2019). While
these studies laid the groundwork for understand-
ing event sequences, durations, and frequencies,
recent work has expanded into implicit or common-
sense dimensions of temporal reasoning.
TORQUE (Ning et al., 2020) is a benchmark de-
signed for answering temporal ordering questions,
while TRACIE, along with its associated model
SYMTIME (Zhou et al., 2021), primarily ensures
temporal-logical consistency rather than modeling
truth probabilities.
McTACO (Zhou et al., 2019) evaluates temporal
commonsense across five dimensions: event dura-
tion, ordering, frequency, stationarity, and typical
time of occurrence. McTACO assesses whether a
given statement aligns with general commonsense
expectations, and does not quantify the likelihood
of a statement being true over time.
Recent work Wenzel and Jatowt, 2023; Jain et al.,
2023 has explored how LLMs handle temporal
commonsense, exposing inconsistencies in event
sequencing and continuity. However, these studies
do not incorporate probabilistic modeling of tem-
poral validity - a core focus of Chronocept, which
models truthfulness as a dynamic, evolving proba-
bility distribution.
2.3 Dataset Structuring for Temporal
Benchmarks
Temporal annotation frameworks like TimeML
(Pustejovsky et al., 2003) and ISO-TimeML (Puste-
jovsky et al., 2010) focus on static event rela-
tionships, often suffering from low inter-annotator
agreement due to event duration ambiguities. The
TimeBank series (Pustejovsky, 2003; Cassidy et al.,
2014) and TempEval challenges (Verhagen, 2007,
2010; UzZaman et al., 2012) expanded evaluations
but remained limited in modeling evolving event
validity.
In response, Ning et al. (2018) proposed a multi-
axis annotation scheme that structures events into
eight distinct categories - Main, Intention, Opin-
ion, Hypothetical, Negation, Generic, Static, and
2

Recurrent. Additionally, the scheme prioritizes
event start points over full event intervals, reducing
ambiguity and significantly improving IAA scores.
Chronocept builds on this by refining multi-axis
annotation to model temporal validity, capturing
how information relevance shifts over time through
probabilistic distributions.
2.4 Statistical Modeling of Temporal Data
Using Skewed Normal Distribution
Traditional normal distributions often fail to cap-
ture skewed temporal patterns. The skew-normal
distribution (Azzalini, 1986, 1996) provides a more
flexible alternative by incorporating asymmetry, im-
proving accuracy in modeling time-dependent in-
formation relevance (Schmidt et al., 2017). Chrono-
cept employs this distribution to capture various
temporal behaviors, including gradual decay, peak
relevance, and rapid obsolescence.
3 Chronocept: Task & Benchmark
Design
3.1 Problem Definition
Temporal Validity Prediction (TVP) of Informa-
tion seeks to model how long a factual statement
remains true after it is published.
We formalize Temporal Validity Prediction as a
probabilistic task of modeling information’s rele-
vance as a continuous probability distribution over
time, rather than the binary-or-multiclass settings
common in earlier work.
LetT⊆R≥0denote the time domain, where
t≥0represents the elapsed time since publication
of information i.
Then, we define a binary random variable,
validityi(t)∈ {0,1} (2)
where validityi(t) = 1 indicates that the infor-
mation iis valid at time t, and validityi(t) = 0
otherwise.
Rather than predicting validityi(t)directly, TVP
aims to learn a continuous probability density func-
tionpi(t)
pi(t) =P 
validityi(t) = 1
, pi:T→[0,1](3)
Accordingly, the probability that the statement
remains valid throughout any interval [a, b]⊆Tis
given by
P
∀t∈[a, b],validityi(t) = 1
=Zb
api(t)dt
(4)Crucially, the model does not impose rigid
boundary constraints - such as pi(0) = 1 or mono-
tonic decay - thereby permitting the learned dis-
tribution to capture complex temporal phenomena,
including delayed onset, non-monotonic plateaus,
and intermittent resurgences (Takemura and Tajima,
2012; Almquist and Jatowt, 2019)
3.2 Modeling Temporal Validity
We model the temporal validity of statements using
a probability curve, with likelihood of being valid
on the Y-axis and time since publication on the
X-axis. To reduce ambiguity, sentences are decom-
posed along semantically distinct axes. A skew-
normal distribution on a logarithmic time scale cap-
tures the validity dynamics.
Axes-Based Decomposition. We adopt the multi-
axis annotation scheme of Ning et al. (2018) (MA-
TRES), which partitions each sentence into eight
semantically coherent axes (Main, Intention, Opin-
ion, Hypothetical, Generic, Negation, Static, Recur-
rent). By isolating relation annotation within each
axis, MATRES reduces cross-category ambiguity
and better aligns with human temporal perception.
In our ablation Appendix F, removing axis
features increases MSE by 4.57%, confirming that
axis-level signals are essential for precise temporal
modeling.
Skewed Normal Distribution. We model tem-
poral validity using the skewed normal distribution,
a generalization of the Gaussian with a shape pa-
rameter αthat captures asymmetry. This enables
representation of non-symmetric temporal patterns
such as delayed onset, gradual decay, or skewed
relevance, which symmetric (Gaussian) or memo-
ryless (exponential) distributions fail to model.
The probability density function is:
f(x;ξ, ω, α ) =2
ωϕx−ξ
ω
Φ
αx−ξ
ω
(5)
where:
•ϕ(z)is the standard normal PDF,
•Φ(z)is the standard normal CDF,
•ξis the location parameter - determining the
time at which an event is most likely valid,
•ωis the scale parameter - governing the dura-
tion of validity, and
3

•αis the shape parameter - controlling skew-
ness (with positive values yielding right skew
and negative values left skew).
Quantitative comparisons against Gaussian,
log-normal, exponential and gamma distributions
in Appendix D support this choice.
Logarithmic Time Scale. Linear time yields
sparse coverage over key intervals, particularly at
minute-level granularity. To address this, we com-
press the time axis using a monotonic logarithmic
transformation:
t′= log1.1(t) (6)
We default to a base of 1.1for the near-
linear spacing across canonical intervals (e.g., min-
utes, days, decades) while preserving granularity.
Chronocept’s target values remain compatible with
alternative bases. See Appendix C for the base
transformation framework, compression analysis,
and the provided code implementation.
4 Dataset Creation
4.1 Benchmark Generation & Pre-Filtering
Chronocept comprises two benchmarks to facili-
tate evaluation across varying complexity levels.
Benchmark I consists of 1,254 samples featuring
simple, single-sentence texts with clear temporal re-
lations - ideal for baseline reasoning - while Bench-
mark II includes 524 samples with complex, multi-
sentence texts capturing nuanced, interdependent
temporal phenomena.
Synthetic samples were generated using the GPT-
o11model (OpenAI, 2024) with tailored prompts
to ensure temporal diversity across benchmarks.
Full prompts for both benchmarks are disclosed
in Appendix E for reproducibility. No real-world
or personally-identifying data was used, ensuring
complete privacy.
In the pre-annotation phase, SBERT2(Reimers
and Gurevych, 2019) and TF-IDF embeddings
were generated for all samples, and pairwise cosine
similarities were calculated. Samples with SBERT
or TF-IDF similarity exceeding 0.7 (70%) were re-
moved to reduce semantic and lexical redundancy.
Annotation guidelines are disclosed in Ap-
pendix A and were continuously accessible during
annotation.
1https://openai.com/o1
2all-MiniLM-L6-v2 available at https://huggingface.
co/sentence-transformers/all-MiniLM-L6-v24.2 Annotation Workflow
Annotation Process. Our protocol consists of
three steps: (i) Temporal Segmentation – partition-
ing text into coherent subtexts that preserve tem-
poral markers; (ii) Axis Categorization – assigning
each segment to one of eight temporal axes (Main,
Intention, Opinion, Hypothetical, Generic, Nega-
tion, Static, Recurrent); and (iii) Temporal Validity
Distribution Plotting – annotating a skewed normal
distribution, parameterized by location ( ξ), scale
(ω), and skewness ( α), over a logarithmic time axis.
To ensure interpretability and consistency, all
parent texts are written in the present tense, dis-
tributions are anchored at t= 0, and multimodal
curves are excluded. Additionally, any samples
that did not exhibit a clearly assignable main time-
line or violated these constraints were flagged and
discarded during the annotation process.
4.3 Annotator Training & Quality Control
Eight third-year B.Tech. students with relevant
coursework in Natural Language Processing, Ma-
chine Learning, and Information Retrieval partic-
ipated. They underwent a 1-hour training session
and a supervised warm-up on 50 samples. Agree-
ment thresholds were set at ICC > 0.90 for numeri-
cal annotations, Jaccard Index > 0.75 for segment-
level annotations, and Pk< 0.15 for segmentation
consistency during this warm-up phase.
Each sample was annotated independently by
two annotators. Quality control included daily re-
views of 10% of annotations, a limit of 70 sam-
ples per annotator per day to mitigate fatigue, and
automated flagging of samples with segmentation
mismatches, target deviations >2 σ, or P k> 0.2.
Discrepancies were adjudicated or, if unresolved,
discarded.
No personal or identifying information was col-
lected or stored during the annotation process.
Handling Edge Cases and Final Resolution.
Ambiguous samples were flagged or discarded
following the three-phase filtering scheme. For
segmentation and axis labels, a union-based
approach retained all plausible interpretations,
recognizing that axis confusion may encode
aspects of human temporal cognition useful for
future modeling. For temporal validity targets
(ξ,ω,α), annotator values were averaged to
yield smooth probabilistic supervision rather than
discrete target selection.
4

{ 
    “_id”: “H0028”,
    “parent_text”: “They are discussing a philosophical concept, whereas an online forum 
simultaneously erupts in debate over similar ideas. They believe open dialogue fosters 
clarity, yet they recognize tensions may escalate. They intend to document their 
conclusions, hoping to contribute thoughtfully to the discussion.”
    “axes”: {
        “main_outcome_axis”: “They are discussing a philosophical concept,”,
        “intention_axis”: “They intend to document their conclusions, hoping to 
contribute thoughtfully to the discussion.”,
        “opinion_axis”: “They believe open dialogue fosters clarity,”,
        “hypothesis_axis”: “”,
        “generic_axis”: “”,
        “negation_axis”: “”,
        “static_axis”: “whereas an online forum simultaneously erupts in debate over 
similar ideas. yet they recognize tensions may escalate.”,
        “recurrent_axis”: “”
    },
    “target_values”: {
        “location”: 39.865,
        “scale”: 13.265,
        “skewness”: 4.25
    }
}Figure 1: Composition of samples in Chronocept benchmarks.
4.4 Inter-Annotator Agreement (IAA)
We evaluate Inter-Annotator Agreement (IAA) us-
ing stage-specific metrics aligned with each step
of the annotation task. Segmentation quality is as-
sessed using the Pkmetric (Beeferman et al., 1997),
axis categorization consistency is measured using
the Jaccard Index, and agreement on the final tem-
poral validity parameters ( ξ,ω,α) is quantified
using the Intraclass Correlation Coefficient (ICC).
We report only ICC as the benchmark-wide
IAA, refraining from aggregating agreement across
stages, as segmentation and axis categorization,
while enriching the dataset structure, do not di-
rectly impact the core prediction task, which de-
pends solely on the parent text and its annotated
temporal validity distribution.
Agreement statistics across both benchmarks are
summarized in Table 1. We observed notable con-
fusion between the Generic andStatic axes during
the early stages of annotation, particularly in the
warm-up phase. This source of disagreement is
analyzed in detail in Appendix B.IAA Metric BI BII
ICC 0.843 0.893
Jaccard Index 0.624 0.731
PkMetric 0.233 0.009
Table 1: IAA metrics for segmentation, axis catego-
rization, and temporal validity annotation across both
benchmarks. For Pk, lower is better, with values rang-
ing from 0 (perfect agreement) to 1 (chance-level).
4.5 Dataset Design
Each Chronocept sample captures the temporal dy-
namics of factual information through a structured
annotation format, as illustrated in Figure 1.
Parent Text. A single sentence serving as the
basis for annotation.
Temporal Axes. Each parent text is segmented
into subtexts annotated along eight temporal axes:
•Main: Core verifiable events.
•Intention: Future plans or desires.
•Opinion: Subjective viewpoints.
5

•Hypothetical: Conditional or imagined
events.
•Negation: Denied or unfulfilled events.
•Generic: Timeless truths or habitual patterns.
•Static: Unchanging states in context.
•Recurrent: Repeated temporal patterns.
Target Values. Temporal validity is quantified
by three parameters:
•ξ(Location): The time point of peak validity.
•ω(Scale): The duration over which validity is
maintained.
•α(Skewness): The asymmetry of the validity
curve.
4.6 Dataset Statistics & Splits
Stratified sampling over the axes distribution was
applied to partition the datasets into training (70%),
validation (20%), and test (10%) splits, ensuring
equitable axis coverage. Table 2 summarizes the
splits for both benchmarks. The axes distribution,
calculated based on non-null annotations for each
sample, is detailed in Table 3.
Benchmark Training Validation Test
Benchmark I 878 247 129
Benchmark II 365 104 55
Table 2: Dataset Composition and Splits.
Temporal Axis Benchmark I Benchmark II
Main Axis 1254 524
Static Axis 516 513
Generic Axis 228 116
Hypothetical Axis 136 182
Negation Axis 240 200
Intention Axis 165 522
Opinion Axis 328 519
Recurrent Axis 348 198
Table 3: Distribution of annotated temporal axes across
Benchmark I and Benchmark II.
Token-level3and target parameter-level statistics
for both benchmarks are summarized in Table 4
and Table 5.
3Tokenization performed using SpaCy’s en_core_web_sm
model: https://spacy.io/api/tokenizerBenchmark Mean Length ( µ) SD ( σ)
Benchmark I 16.41 tokens 1.56 tokens
Benchmark II 56.21 tokens 6.21 tokens
Table 4: Sentence Length Statistics for Benchmarks.
4.7 Accessibility and Licensing
The Chronocept dataset is released under the Cre-
ative Commons Attribution 4.0 International (CC-
BY 4.0)4license, allowing unrestricted use with
proper attribution. It is publicly available on Hug-
ging Face Datasets at: https://huggingface.
co/datasets/krishgoel/chronocept .
5 Baseline Model Performance
5.1 Task Scope and Evaluation Focus
Chronocept models temporal validity as a struc-
tured regression task over low-dimensional param-
eters: location ( ξ), scale ( ω), and skewness ( α),
predicted from annotated parent texts. Unlike prior
work on event ordering (Pustejovsky, 2003), com-
monsense classification (Zhou et al., 2019), or tem-
poral shift detection (Wenzel and Jatowt, 2024),
segmentation and axis labels are treated as prepro-
cessing and not modeled at inference.
Evaluation spans three dimensions: regression
accuracy (MSE, MAE, R2), calibration (Negative
Log-Likelihood), and rank correlation (Spearman
ρ). As the task involves parameter estimation rather
than text generation, encoder-only models suffice.
Decoder architectures are unnecessary, as Chrono-
cept operates at the application layer, interfacing
with downstream systems without altering core lan-
guage models.
5.2 Baseline Models and Training Setup
We benchmark Chronocept against a representative
set of baselines spanning statistical (LR, SVR), tree-
based (XGB), and neural architectures (FFNN, Bi-
LSTM, BERT Regressor). Each baseline is trained
to jointly predict ξ,ωandαfrom BERT-based in-
put embeddings of the parent text and temporal
subtexts. Targets are Z-Score normalized to stan-
dardize learning across all models.
Hyperparameters for all baselines (except BERT)
were tuned via grid search; final configurations
are detailed in Appendix H. FFNN and Bi-LSTM
models were trained for 100 epochs while BERT
4https://creativecommons.org/licenses/by/4.0
6

Parameter Location ( ξ) Duration ( ω) Skewness ( α)
Benchmark Mean ( µ) SD ( σ) Mean ( µ) SD ( σ) Mean ( µ) SD ( σ)
Benchmark I 54.2803 20.4169 11.5474 3.7725 -0.0158 1.3858
Benchmark II 46.1511 13.3839 9.5553 2.5725 0.0275 1.1773
Table 5: Temporal Parameter Distribution Statistics for Benchmarks.
0 10 20 30 40 50
Epochs50100150200250LossBERT Regressor Loss Curves for Both Benchmarks
Train Loss (Benchmark 1)
Validation Loss (Benchmark 1)
Train Loss (Benchmark 2)
Validation Loss (Benchmark 2)
Figure 2: BERT training loss curves for Benchmark I
and Benchmark II. The loss flatlined after 2 epochs for
both benchmarks.
was trained for 50 epochs. BERT training loss
plateaued after approximately 2 epochs across both
benchmarks, as shown in Figure 2, suggesting early
stopping could be beneficial for future experiments.
All training and inference experiments were con-
ducted on a machine with an Intel Core i9-14900K
CPU, 16GB DDR5 RAM, and an NVIDIA RTX
4060 GPU.
Baseline implementations and training scripts
are publicly available at: https://github.com/
krishgoel/chronocept-baseline-models .
5.3 Quantitative Evaluation
Table 6 summarizes the performance of baseline
models across both benchmarks. Each reported
metric reflects the mean score across the three pre-
dicted parameters.
Feedforward Neural Networks (FFNN) outper-
form all other models overall, achieving the lowest
MSE, MAE, NLL, and the highest Spearman Cor-
relation on Benchmark I. This supports prior find-
ings that simpler architectures, when paired with
high-quality pretrained embeddings, can match or
exceed deeper models in accuracy and efficiency
(Saphra and Lopez, 2019; Wei et al., 2021).
Bi-LSTM trails FFNN on Benchmark I but out-
performs it on Benchmark II in four of five metrics -
MSE, R2, NLL and Spearman ρ- on Benchmark II,which provides longer textual context. This is con-
sistent with prior findings on sequence modeling
(Meng and Rumshisky, 2018; Dligach et al., 2017),
and may stem from Bi-LSTM’s ability to better
model long-range dependencies, while FFNNs rely
on the BERT [CLS] token, which can struggle to
encode longer contexts into a single vector (Li et al.,
2020).
BERT Regression improves significantly from
Benchmark I to II, with MSE dropping by over
50%, suggesting longer inputs help stabilize fine-
tuning. However, BERT still underperforms across
all metrics, consistent with its known sensitivity to
overfitting and gradient noise on small regression
datasets (Mosbach et al., 2021; Peters et al., 2019;
Lee et al., 2020).
Among classical models, SVR and XGBoost
perform reasonably but are outpaced by neural ap-
proaches. SVR achieves relatively strong R2and
NLL scores on Benchmark I, while XGBoost and
LR lag across all metrics. Their interpretability
and training efficiency still make them useful refer-
ence baselines (Drucker et al., 1996; Rogers et al.,
2020).
Together, these results affirm that pretrained
embeddings paired with compact neural regres-
sors like FFNN yield state-of-the-art performance.
Additionally, they highlight how models with
sequence-awareness, such as Bi-LSTM and BERT,
benefit disproportionately from longer contexts.
5.4 Impact of Temporal Axes: Ablation
Studies
To assess the utility of explicit temporal axes in
Chronocept, we conduct two ablation studies on
Benchmark 1 using the Bi-LSTM and FFNN base-
lines.
The first study evaluates the impact of removing
all axis-level information, and the second examines
the impact of randomly shuffling axis order during
training. This setup parallels prior work on robust-
ness testing via perturbed input labels (Moradi and
Samwald, 2021).
7

Metric MSE MAE R2NLL Spearman
Baseline BI BII BI BII BI BII BI BII BI BII
LR 1.3610 1.1009 0.9179 0.8361 -0.3610 -0.1009 1.5730 1.4670 0.2338 0.3279
XGB 0.8884 0.9580 0.7424 0.8011 0.1116 0.0420 1.3598 1.3975 0.2940 0.2331
SVR 0.9067 0.8889 0.7529 0.7740 0.0933 0.1111 1.3700 1.3601 0.3281 0.3293
FFNN 0.8763 0.8715 0.7284 0.7583 0.1237 0.1285 1.3529 1.3502 0.3543 0.3437
Bi-LSTM 0.9203 0.8702 0.7571 0.7646 0.0797 0.1298 1.3774 1.3494 0.2367 0.3535
BERT 145.8611 68.1507 6.7570 4.6741 -0.0090 -0.1122 3.9103 3.5299 -0.0485 -0.2407
Table 6: Test set performance of baseline models for Benchmark I (BI) and Benchmark II (BII). Lower values for
MSE, MAE, and NLL indicate better performance; higher R2and Spearman Correlation ρdenote improved fit.
Both the axis -removal and axis -shuffle setups
lead to substantial performance degradation, indi-
cating that both - the presence and consistent order-
ing of temporal axes - play a key role in accurately
modeling temporal validity.
Table 7 summarizes the increase in MSE for
the Bi-LSTM baseline. Experimental design and
complete results for both baselines are detailed
in Appendix F (excluded axes) and Appendix G
(shuffled axes).
Ablation Type Ablated MSE Increase
Exclusion of Axes 0.9625 4.59%
Erroneous Labeling 1.0107 9.83%
Table 7: Ablation results for the Bi-LSTM baseline.
Relative increases are computed over the original MSE
of0.9203 .
6 Conclusion & Applications
We introduced Chronocept, a framework that mod-
els temporal validity as a continuous probability
distribution using a unified, parameterized repre-
sentation. By encoding validity through location
(ξ), scale ( ω), and skewness ( α), Chronocept pro-
vides a generalizable mathematical scheme for tem-
poral reasoning in language.
Through structured annotations and explicit tem-
poral axes, Chronocept enables models to capture
not just if, but when and for how long informa-
tion remains valid - advancing beyond binary truth
labels to a richer temporal understanding.
Empirical results highlight the effectiveness of
simple neural models paired with pretrained em-
beddings, and ablation studies underscore the im-
portance of structural consistency and axis-level
decomposition.
Chronocept opens pathways for temporallyaware applications, including retrieval-augmented
generation (RAG), fact verification, knowledge life-
cycle modeling, and proactive AI agents that act
based on temporal salience (Miksik et al., 2020).
All datasets, annotations, and baselines are pub-
licly released to support continued research in this
space.
7 Limitations
In this section, we highlight key limitations of
Chronocept and suggest directions for future re-
finement and broader applicability.
Unimodal Temporal Representation. Chrono-
cept models temporal validity with a unimodal,
single-peaked distribution. While this ensures in-
terpretability and efficient annotation, it cannot rep-
resent events with multiple distinct periods of rele-
vance, such as seasonal or recurring phenomena.
Sentence-Level Context Only. The dataset con-
sists of short, self-contained sentences without
document-level or historical context. This limits
the modeling of long-range temporal dependencies
and evolving narratives, constraining discourse-
level temporal reasoning.
No Atemporality Indicators. Chronocept lacks
explicit labels for atemporal or universally valid
facts, introducing ambiguity between permanently
valid and time-sensitive information.
Minimum Validity Constraint from Log Time
Scale. The logarithmic time scale imposes a
lower bound of one minute, making it unsuitable
for modeling events that become instantly obsolete,
such as flash updates or ephemeral statements.
8 Acknowledgments
We thank Mohammed Iqbal, Meenakshi Kumar,
Yudhajit Mondal, Tanish Sharma, Devansh Sharma,
8

Lakshya Paliwal, Ishaan Verma, and Sanjit Chitturi
for their help with data annotation.
References
James F Allen. 1983. Maintaining knowledge about
temporal intervals. Commun. ACM , 26(11):832–843.
Axel Almquist and Adam Jatowt. 2019. Towards con-
tent expiry date determination: Predicting validity
periods of sentences. pages 86–101.
A Azzalini. 1996. The multivariate skew-normal distri-
bution. Biometrika , 83(4):715–726.
Adelchi Azzalini. 1986. A class of distributions which
includes the normal ones. Scandinavian Journal of
Statistics .
Doug Beeferman, Adam Berger, and John Lafferty.
1997. Text segmentation using exponential mod-
els. In Second Conference on Empirical Methods
in Natural Language Processing .
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D. Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Gaurav Sastry,
Amanda Askell, Ariel Agarwal, Shelly Herbert-V oss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens
Winter, Christopher Hesse, Mark Chen, Eric Sigler,
Mateusz Litwin, Scott Gray, Benjamin Chess, Jack
Clark, Christopher Berner, Sam McCandlish, Alec
Radford, Ilya Sutskever, and Dario Amodei. 2020.
Language models are few-shot learners. 33:1877–
1901.
Taylor Cassidy, Bill McDowell, Nathanael Chambers,
and Steven Bethard. 2014. An annotation framework
for dense event ordering.
Supratim Das, Arunav Mishra, Klaus Berberich, and
Vinay Setty. 2017. Estimating event focus time using
neural word embeddings. In Proceedings of the 2017
ACM on Conference on Information and Knowledge
Management , New York, NY , USA. ACM.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. Bert: Pre-training of deep
bidirectional transformers for language understand-
ing. pages 4171–4186.
Dmitriy Dligach, Timothy Miller, Chen Lin, Steven
Bethard, and Guergana Savova. 2017. Neural tem-
poral relation extraction. In Proceedings of the 15th
Conference of the European Chapter of the Associa-
tion for Computational Linguistics: Volume 2, Short
Papers , pages 746–751, Valencia, Spain. Association
for Computational Linguistics.
Harris Drucker, Christopher J. C. Burges, Linda Kauf-
man, Alex Smola, and Vladimir Vapnik. 1996. Sup-
port vector regression machines. 9.Rhailana Fontes, Jéssica Ribeiro, Daya S Gupta, Dionis
Machado, Fernando Lopes-Júnior, Francisco Magal-
hães, Victor Hugo Bastos, Kaline Rocha, Victor Mar-
inho, Gildário Lima, Bruna Velasques, Pedro Ribeiro,
Marco Orsini, Bruno Pessoa, Marco Antonio Araujo
Leite, and Silmar Teixeira. 2016. Time perception
mechanisms at central nervous system. Neurol. Int. ,
8(1):5939.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian
Sun. 2016. Deep residual learning for image recogni-
tion. pages 770–778.
Taishi Hosokawa, Adam Jatowt, and Kazunari
Sugiyama. 2023. Temporal natural language infer-
ence: Evidence-based evaluation of temporal text va-
lidity. In Lecture Notes in Computer Science , Lecture
notes in computer science, pages 441–458. Springer
Nature Switzerland, Cham.
Raghav Jain, Daivik Sojitra, Arkadeep Acharya, Sri-
parna Saha, Adam Jatowt, and Sandipan Dandapat.
2023. Do language models have a common sense
regarding time? revisiting temporal commonsense
reasoning in the era of large language models. In Pro-
ceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing , pages 6750–
6774, Stroudsburg, PA, USA. Association for Com-
putational Linguistics.
Adam Jatowt, Ching-Man Au Yeung, and Katsumi
Tanaka. 2013. Estimating document focus time. In
Proceedings of the 22nd ACM international confer-
ence on Conference on information & knowledge
management - CIKM ’13 , New York, New York,
USA. ACM Press.
Nattiya Kanhabua and Kjetil Nørvåg. 2008. Improv-
ing temporal language models for determining time
of non-timestamped documents. In Research and
Advanced Technology for Digital Libraries , Lecture
notes in computer science, pages 358–370. Springer
Berlin Heidelberg, Berlin, Heidelberg.
Abhimanu Kumar, Jason Baldridge, Matthew Lease,
and Joydeep Ghosh. 2012. Dating texts without ex-
plicit temporal cues. arXiv [cs.CL] .
Brenden M. Lake and Marco Baroni. 2018. General-
ization without systematicity: On the compositional
skills of sequence-to-sequence recurrent networks.
Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon
Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang.
2020. Biobert: a pre-trained biomedical language
representation model for biomedical text mining.
Bioinformatics , 36(4):1234–1240.
Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter
Abbeel. 2016. End-to-end training of deep visuomo-
tor policies. Journal of Machine Learning Research ,
17(39):1–40.
Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang,
Yiming Yang, and Lei Li. 2020. On the sentence
embeddings from pre-trained language models. In
9

Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP) ,
pages 9119–9130, Online. Association for Computa-
tional Linguistics.
Kelvin Luu, Daniel Khashabi, Suchin Gururangan, Kar-
ishma Mandyam, and Noah A Smith. 2021. Time
waits for no one! analysis and challenges of temporal
misalignment. arXiv [cs.CL] .
Steven Lynden, Mehari Heilemariam, Kyoung-Sook
Kim, Adam Jatowt, Akiyoshi Matono, Hai-Tao Yu,
Xin Liu, and Yijun Duan. 2023. Commonsense tem-
poral action knowledge (cotak) dataset. In Proceed-
ings of the 32nd ACM International Conference on In-
formation and Knowledge Management (CIKM 2023 .
Yuanliang Meng and Anna Rumshisky. 2018. Context-
aware neural model for temporal information extrac-
tion. In Proceedings of the 56th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 527–536, Melbourne,
Australia. Association for Computational Linguistics.
Ondrej Miksik, I Munasinghe, J Asensio-Cubero,
S Reddy Bethi, ST Huang, S Zylfo, Xuechen Liu,
T Nica, A Mitrocsak, S Mezza, et al. 2020. Building
proactive voice assistants: When and how (not) to
interact. arXiv preprint arXiv:2005.01322 .
Milad Moradi and Matthias Samwald. 2021. Evaluating
the robustness of neural language models to input
perturbations. In Proceedings of the 2021 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing , pages 1558–1570, Online and Punta Cana,
Dominican Republic. Association for Computational
Linguistics.
Marius Mosbach, Maksym Andriushchenko, and Diet-
rich Klakow. 2021. On the stability of fine-tuning
bert: Misconceptions, explanations, and strong base-
lines. In Proceedings of the 9th International Confer-
ence on Learning Representations (ICLR) .
Qiang Ning, Hao Wu, Rujun Han, Nanyun Peng, Matt
Gardner, and Dan Roth. 2020. TORQUE: A reading
comprehension dataset of temporal ordering ques-
tions. In Proceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing
(EMNLP) , Stroudsburg, PA, USA. Association for
Computational Linguistics.
Qiang Ning, Hao Wu, and Dan Roth. 2018. A multi-
axis annotation scheme for event temporal relations.
InProceedings of the 56th Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers) , pages 1318–1328. Association for
Computational Linguistics.
OpenAI. 2024. Openai o1 system card. arXiv .
Matthew E. Peters, Sebastian Ruder, and Noah A. Smith.
2019. To tune or not to tune? adapting pretrained
representations to diverse tasks. In Proceedings of
the 4th Workshop on Representation Learning for
NLP (RepL4NLP-2019) , pages 7–14, Florence, Italy.
Association for Computational Linguistics.J Pustejovsky, Kiyong Lee, H Bunt, and Laurent Ro-
mary. 2010. ISO-TimeML: An international standard
for semantic annotation. LREC , pages 394–397.
James Pustejovsky. 2003. The timebank corpus. Corpus
linguistics .
James Pustejovsky, José M Castaño, Robert Ingria, and
Graham Katz. 2003. TimeML: A specification lan-
guage for temporal and event expressions.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using siamese BERT-
networks. arXiv [cs.CL] .
Anna Rogers, Olga Kovaleva, and Anna Rumshisky.
2020. A primer in BERTology: What we know about
how BERT works. Transactions of the Association
for Computational Linguistics , 8:842–866.
Naomi Saphra and Adam Lopez. 2019. Understanding
learning dynamics of language models with SVCCA.
InProceedings of the 2019 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies,
Volume 1 (Long and Short Papers) , pages 3257–3267,
Minneapolis, Minnesota. Association for Computa-
tional Linguistics.
Alexandra M Schmidt, Kelly C M Gonçalves, and Pa-
trícia L Velozo. 2017. Spatiotemporal models for
skewed processes. Environmetrics , 28(6):e2411.
Anders Søgaard and Yoav Goldberg. 2016. Deep multi-
task learning with low level tasks supervised at lower
layers. pages 231–235.
Hikaru Takemura and Keishi Tajima. 2012. Tweet clas-
sification based on their lifetime duration.
Naushad UzZaman, Hector Llorens, James Allen, Leon
Derczynski, Marc Verhagen, and James Pustejovsky.
2012. TempEval-3: Evaluating events, time expres-
sions, and temporal relations. arXiv [cs.CL] .
Marc Verhagen. 2007. Semeval-2007 task 15: Tempe-
val temporal relation identification. In Proceedings
of the fourth international workshop on semantic
evaluations .
Marc Verhagen. 2010. SemEval-2010 task 13:
TempEval-2. In Proceedings of the 5th international
workshop on semantic evaluation .
Colin Wei, Sang Michael Xie, and Tengyu Ma. 2021.
Why do pretrained language models help in down-
stream tasks? an analysis of head and prompt tuning.
Advances in Neural Information Processing Systems ,
34:16158–16170.
Haoyang Wen and Heng Ji. 2021. Utilizing relative
event time to enhance event-event temporal relation
extraction. In Proceedings of the 2021 Conference on
Empirical Methods in Natural Language Processing ,
Stroudsburg, PA, USA. Association for Computa-
tional Linguistics.
10

Georg Wenzel and Adam Jatowt. 2023. An overview
of temporal commonsense reasoning and acquisition.
arXiv [cs.AI] .
Georg Wenzel and Adam Jatowt. 2024. Temporal va-
lidity change prediction. In Findings of the Asso-
ciation for Computational Linguistics: ACL 2024 ,
pages 1424–1446, Bangkok, Thailand. Association
for Computational Linguistics.
Ben Zhou, Daniel Khashabi, Qiang Ning, and Dan Roth.
2019. “going on a vacation” takes longer than “go-
ing for a walk”: A study of temporal commonsense
understanding. In Proceedings of the 2019 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing and the 9th International Joint Conference
on Natural Language Processing (EMNLP-IJCNLP) ,
pages 3363–3369, Stroudsburg, PA, USA. Associa-
tion for Computational Linguistics.
Appendix
A Annotation Guidelines
This section outlines the annotation guidelines used
in the Chronocept dataset. These were introduced
through an in-person training session and remained
accessible throughout the annotation phase via a
custom Streamlit-based interface for annotations5.
The guidelines provide precise instructions for tem-
poral segmentation, axis categorization, and tem-
poral validity distribution plotting, supplemented
with definitions, examples, and coverage of edge
cases for all eight temporal axes.
During the initial warm-up phase, annotators ex-
hibited substantial confusion between the Generic
and Static axes. To mitigate this, the guidelines
were revised to incorporate clearer contextual defi-
nitions and axis-specific "key questions" designed
to improve disambiguation. These revisions led to a
marked improvement in inter annotator agreement.
The complete guidelines are shown in Figure 3.
5https://streamlit.ioB Axis Confusion Analysis: Generic and
Static
Int entionOpinionHypo .GenericSt atic R ecurr ent NegationInt entionOpinionHypo .Generic St atic R ecurr ent Negation
(a) Axis assignment co-occurrence matrix with Generic and
Static treated as distinct classes
Int entionOpinionHypo .St atic +
Generic R ecurr entNegationInt entionOpinionHypo .St atic +
Generic R ecurr entNegation
(b) Axis assignment co-occurrence matrix after merging
Generic and Static into a unified class
Figure 4: Comparison of co-occurrence matrices before
and after merging the Generic and Static axes, used to
assess annotation consistency.
This appendix investigates a key source of anno-
tator disagreement in the Chronocept annotation
process: the difficulty in consistently distinguish-
ing between the Generic and Static temporal axes.
Generic segments typically express habitual or
timeless statements, while Static segments describe
ongoing but context-specific states. Their seman-
tic similarity led to frequent disagreement in axis
11

assignment.
To address this, the annotation guidelines were
refined during the warm-up phase with axis-
specific clarifications and diagnostic questions.
The guideline clarification led to reduced confu-
sion, as shown in the co-occurrence matrices in
Figure 4.
While co-occurrence matrices are traditionally
used to analyze disagreement patterns between an-
notators, we treat them here as confusion matrices
by including agreement counts along the diagonal,
enabling standard metric computation.
To quantify the benefit of merging these axes,
we computed micro-averaged inter-annotator pre-
cision. Treating this as a multi-class classification
task, we additionally calculate Cohen’s Kappa to
assess inter-annotator agreement beyond chance.
As shown in Table 8, merging resulted in a consis-
tent improvement across both metrics: precision
improved by 18.0% and Cohen’s Kappa by 17.47%.
Axis Setting Precision Cohen’s Kappa
Original 0.4443 0.3291
Merged 0.5243 0.3866
Table 8: Improvement in annotator alignment metrics
after merging Generic and Static into a single class.
CTime Scale Logarithm Base Conversion
1 minute 1 hour 1 day 1 week 1 month 1 year 1 decade
Timestamp (t) [minutes]020406080100120140160Logarithmic Scale Value (t')
Logarithmic Scale Transformations with Various Bases
log base 1.10
log base 2.00
log base 2.72
log base 10.00
Figure 5: Effect of logarithmic base choice on time axis
representation. Base 1.1 preserves quasi-linear spacing;
larger bases induce stronger compression.
Chronocept represents time on a logarithmic axis to
unify short- and long-term temporal dynamics in a
compact space. The transformation is defined over
a configurable base b; all released datasets use base
1.1. A reusable DataLoader with log conversionis available in the official baselines repository6.
Log Transformation. Given time tin minutes,
the log-space representation is:
t′=ln(t)
ln(b).
Base 1.1yields quasi-linear spacing across inter-
vals like hours, days, and years, preserving inter-
pretability. Figure 5 shows that higher bases in-
creasingly compress longer intervals, while base
1.1maintains resolution across scales.
Compression Analysis. Table 9 summarizes the
compression effect across bases 1.1,2, and10. For
each timestamp, we report the log value t′, com-
pression ratio CR=t′/t, and percentage compres-
sion.
To convert values between log bases mandb:
t′(b)=ln(m)
ln(b)·t′(m).
Skew-Normal Parameter Adjustment. Chrono-
cept models temporal validity using a skew-normal
distribution:
f(x;ξ, ω, α ) =2
ωϕx−ξ
ω
Φ
αx−ξ
ω
,
where ξandωdenote location and scale. When
converting between bases:
ξ(b)=ln(m)
ln(b)·ξ(m), ω(b)=ln(m)
ln(b)·ω(m).
Skewness αremains invariant.
D Comparison of Distributions for
Modeling Temporal Validity and Curve
Fitting Methodology
This section evaluates candidate distributions for
modeling temporal validity and outlines the curve
fitting methodology. We consider six synthetic,
unimodal scenarios varying along three axes: offset
(peak position), duration (span of validity), and
asymmetry (skew in rise and decay). Table 10 lists
a representative sentence and five annotation points
per scenario, placed on a base-1.1 logarithmic time
axis.
Each temporal profile is defined by a smooth
freehand curve from which five points are sam-
pled—one at the peak, two mid-validity, and two
6https://github.com/krishgoel/
chronocept-baseline-models
12

log base 1.1 log base 2 log base 10
Timestamp Linear (t) t′CR % t′CR % t′CR %
1 minute 1 0.0 0.000 100 0.0 0.000 100 0.0 0.000 100
1 hour 60 42.96 0.716 28.4 5.91 0.099 90.1 1.78 0.030 97.0
1 day 1440 76.30 0.053 94.7 10.47 0.007 99.3 3.16 0.002 99.8
1 week 10080 96.73 0.010 99.0 13.30 0.001 99.9 4.00 3.968e-4 99.9
1 month 43200 111.97 0.003 99.7 15.39 3.563e-4 99.9 4.63 1.072e-4 ~100
1 year 525600 138.23 2.623e-4 ~100 19.00 3.615e-5 ~100 5.72 1.088e-5 ~100
1 decade 5256000 162.25 3.087e-5 ~100 22.33 4.249e-6 ~100 6.72 1.279e-6 ~100
Table 9: Compression analysis across logarithmic bases. CR = t′/t, Compression % = 100×(1−CR).
low-validity points. These define a proportional
shape used for fitting.
Since these curves represent relative probabil-
ities, their area under the curve (AUC) is uncon-
strained. During optimization, a scaling factor is
applied to fit freely, followed by Trapezoidal Rule
normalization to enforce AUC = 1 while preserving
shape.
To reduce computational overhead over long-
tailed domains, we recommend rescaling the fitted
curve by its maximum value to constrain it to [0,1].
This avoids instability from very small values
in AUC-normalized densities. The result, while
no longer a true probability distribution, retains
shape and relative comparisons. We refer to it as a
proportional validity curve , useful in applications
prioritizing ranking or visualization over strict
probabilistic semantics.
Candidate distributions include:
Gaussian Normal:
fGaussian (x;µ, σ) =1√
2π σexp
−(x−µ)2
2σ2
Exponential:
fExp(x;λ) =λexp(−λx),where x≥0
Log-normal:
fLN(x;µ, σ) =1
x√
2π σexp
−(lnx−µ)2
2σ2
,
where x >0
Gamma:
fΓ(x;k, θ) =1
Γ(k)θkxk−1exp
−x
θ
,
where x >0Skewed Normal:
fSN(x;ξ, ω, α ) =2
ωϕx−ξ
ω
Φ
αx−ξ
ω
where ϕ(z)is the standard normal PDF and Φ(z)
is the standard normal CDF.
Optimization: Parameter estimation is performed
using the Trust Region Reflective (TRF) algorithm
by minimizing the sum of squared residuals:
SSR(θ) =NX
i=1(yi−f(xi;θ))2
This is implemented via
scipy.optimize.curve_fit7. After opti-
mization, we compute:
N=Zxmax
xminffit(x)dx,
fnorm(x) =ffit(x)
N, f max= max
xfnorm(x),
Sfinal=Sfit
N·fmax
Evaluation: RMSE is used as the primary
goodness-of-fit metric. As a scale-sensitive mea-
sure that penalizes large deviations, a lower RMSE
indicates superior fit quality.
Table 10 and Figure 6 present the six scenarios,
annotation points, and corresponding fitted curves.
Table 11 reports RMSE for each candidate distri-
bution across scenarios. The skew-normal con-
sistently yields the lowest RMSE, confirming its
suitability for modeling asymmetric and variable-
duration temporal profiles.
7https://docs.scipy.org/doc/scipy/reference/
generated/scipy.optimize.curve_fit.html
13

Temporal Scenario Sample Sentence Annotation Points (x, y)
S1: Early Onset "He is making coffee for
himself right now."(14.91, 0.19), (21.64, 0.41),
(27.64, 0.77), (31.64, 0.41),
(34.91, 0.20)
S2: Late Onset "The movie is going to hit the
theaters in a few weeks."(93.75, 0.21), (100.67, 0.80),
(106.57, 0.42), (112.73, 0.20),
(98.0, 0.63)
S3: Short Duration "The site has been crashing for
a few minutes as there is some
server maintenance work going
on."(12.73, 0.21), (28.19, 0.80),
(41.28, 0.20), (32.19, 0.60),
(18.91, 0.40)
S4: Long Duration "The ruling government brings
growth and progress."(1, 0.05), (130.38, 0.81),
(147.84, 0.21), (111.29, 0.42),
(138.38, 0.60)
S5: Rapid Rise, Slow Decay "The advertisement’s impact
peaks immediately and lingers."(42.73, 0.21), (46.91, 0.40),
(53.10, 0.80), (63.46, 0.56),
(81.83, 0.27)
S6: Slow Rise, Rapid Decay "The news slowly gains
attention but quickly becomes
outdated."(43.28, 0.20), (58.01, 0.40),
(76.92, 0.79), (84.92, 0.40),
(88.92, 0.17)
Table 10: Six temporal scenarios illustrating the effects of offset, duration, and asymmetry. Each scenario is
represented by 5 annotation points on a log-transformed time axis with base 1.1.
E Synthetic Generation of Samples
This section presents the plaintext markdown
prompts used for synthetic dataset generation in
Chronocept via the GPT-o1 model (OpenAI, 2024).
The prompts are designed to yield syntactically
coherent text with explicit temporal structure. Gen-
eration was performed in batches of 50 samples per
prompt.
The prompts are shown in Figure 7 for
Benchmark-I and Figure 8 for Benchmark-II.
F Ablation Study: Impact of Structured
Temporal Axes on Model Performance
To evaluate the contribution of multi-axis temporal
annotations in modeling temporal validity, we con-
duct an ablation study on the Bi-LSTM and FFNN
baselines. Specifically, we assess the effect of re-
moving structured temporal axes from the model
input.
Input Construction. Each example in Chrono-
cept is annotated along multiple temporal axes. In
the standard setup, axis-specific embeddings are
concatenated in a fixed order to the embedding ofthe parent text, forming a structured input repre-
sentation. The ablation removes these axis embed-
dings, retaining only the parent text embedding.
Setup. We compare the two configurations (with
and without axis embeddings) using Bi-LSTM and
FFNN models on Benchmark I. Both models are
trained to predict the parameters ξ,ω, andαof the
skew-normal temporal validity distribution. Eval-
uation is performed using MSE, MAE, R2, NLL,
and CRPS.
Results. Table 12 reports the results for both
models. Including axis embeddings reduces
Bi-LSTM MSE by 4.6% and boosts R2by 112%,
confirming that structured cues matter more for
goodness -of-fit than for absolute error. FFNN sees
a 6.9% MSE drop and a 95.7% gain in R2, exhibit-
ing a similar trend with even greater error reduction
across all metrics.
These findings are consistent with prior work
showing that compositional and auxiliary structure
improves model generalization and fit across tasks
(Lake and Baroni, 2018; Søgaard and Goldberg,
2016).
14

Distribution S1 S2 S3 S4 S5 S6 Parameters
Gaussian 0.0709 0.0673 0.0424 0.0273 0.1193 0.0806 (µ, σ)
Exponential 0.2103 0.2291 0.2312 0.2704 0.2126 0.2212 (λ)
Log-normal 0.0844 0.0597 0.0804 0.0325 0.0872 0.0919 (µ, σ)
Gamma 0.0827 0.0623 0.0668 0.0307 0.0968 0.0899 (k, θ)
Skewed Normal 0.0514 0.0357 0.0407 0.0224 0.0505 0.0247 (ξ, ω, α )
Table 11: Average RMSE values for candidate distributions across six temporal scenarios. All distributions were
fitted using a scaling factor Sto enforce AUC = 1. A lower RMSE indicates a better fit, as RMSE heavily penalizes
large errors due to squaring, is scale-dependent, and more sensitive to outliers.
Model Setting MSE MAE R2NLL CRPS
Bi-LSTMWithout Axes 0.9625 0.7659 0.0375 1.3998 0.7659
Absolute Change ( ∆) 0.0422 0.0088 0.0422 0.0224 0.0088
Improvement 4.59% 1.16% 112.53% 1.63% 1.16%
FFNNWithout Axes 0.9368 0.7531 0.0632 1.3863 0.7531
Absolute Change ( ∆) 0.0605 0.0247 0.0605 0.0334 0.0247
Improvement 6.91% 3.39% 95.71% 2.47% 3.39%
Table 12: Ablation results on Benchmark I for Bi-LSTM and FFNN with axis embeddings removed. “Absolute
Change” rows show differences from the original metrics in Table 6.
Conclusion. Structured axis embeddings im-
prove performance across both architectures, par-
ticularly in R2, which nearly doubles, indicating
better distributional alignment. These results vali-
date Chronocept’s use of explicit temporal structure
and are consistent with prior work on structured
auxiliary signals.
G Ablation Study: Impact of Incorrect
Temporal Axes Labeling
We evaluate the sensitivity of temporal validity
modeling to erroneous axis labelling by conduct-
ing an ablation on FFNN and Bi-LSTM baselines.
Specifically, we shuffle the order of temporal axis
embeddings during training while preserving cor-
rect ordering in the test set.
Setup. In Chronocept, input representations are
formed by concatenating temporal axis embeddings
in a fixed sequence with the parent text embedding.
This ablation introduces erroneous axis labelling by
disrupting the axis order during training, thereby
breaking the structural alignment. The evaluation
set remains unperturbed. Models are trained to
predict skew-normal parameters ξ,ω, and α, and
evaluated on Benchmark I using MSE, MAE, R2,
NLL, and CRPS.Results. Table 13 shows that misaligned axis or-
dering during training degrades performance signif-
icantly. Bi-LSTM MSE increases by 9.81% and R2
decreases by 113.43%; FFNN sees a 13.36% MSE
increase and 94.58% R2decrease. These results
suggest that disrupting structural alignment intro-
duces inductive noise, echoing prior findings on the
role of compositional structure (Lake and Baroni,
2018) and input robustness (Moradi and Samwald,
2021). The pronounced drop in R2highlights that
axis ordering is critical for fit quality.
Conclusion. Erroneous axis labelling during
training leads to statistically significant drops in
performance, particularly in R2, highlighting the
importance of Chronocept’s structured multi-axis
representation for accurate temporal modeling.
H Hyperparameter Search and Final
Baseline Configurations
All baseline models were tuned via grid search on
the validation split of each benchmark. All neural
models except BERT were trained for 100 epochs,
with early stopping applied based on validation loss
when applicable. BERT was trained for 50 epochs.
Final hyperparameters are summarized below.
Support Vector Regression (SVR). We
searched over C∈ {0.1,1,10},ε∈ {0.01,0.1,1},
15

Model Setting MSE MAE R2NLL CRPS
Bi-LSTMErroneous Axes 1.0107 0.7984 -0.0107 1.4243 0.7984
Absolute Change ( ∆) 0.0904 0.0413 −0.0904 0.0469 0.0413
Performance Drop 9.81% 5.46% 113.43% 3.40% 5.46%
FFNNErroneous Axes 0.9933 0.7591 0.0067 1.4156 0.7591
Absolute Change ( ∆) 0.1170 0.0307 −0.1170 0.0627 0.0307
Performance Drop 13.36% 4.21% 94.58% 4.63% 4.21%
Table 13: Ablation results on Benchmark I for Bi-LSTM and FFNN under erroneous temporal axis labelling during
training. “Absolute Change” rows show differences from the original metrics in Table 6.
and kernel type ∈ {linear ,rbf}. The optimal
setting used an RBF kernel with C= 1andε= 1
(see Table 14).
Benchmark C ε Kernel
Benchmark I 1 1 rbf
Benchmark II 1 1 rbf
Table 14: Final SVR hyperparameters.
Linear Regression (LR). The grid search over
fit_intercept ∈ {True,False}selected False in
both cases (see Table 15).
Benchmark Fit Intercept
Benchmark I False
Benchmark II False
Table 15: Final Linear Regression setting.
XGBoost (XGB). We tuned n_estimators ∈
{50,100},max _depth ∈ {3,5}, and learning rate
∈ {0.1,0.01}. The best configuration used 50 es-
timators, depth 3, and learning rate 0.1 (see Ta-
ble 16).
Benchmark n Depth Learning Rate
Benchmark I 50 3 0.1
Benchmark II 50 3 0.1
Table 16: Final XGBoost hyperparameters.
Feedforward Neural Network (FFNN). We
searched over hidden size ∈ { 64,128,256},
dropout ∈ { 0.0,0.2,0.5}, learning rate
∈ { 0.01,0.001,0.0001}, L1 regularization
∈ { 0.0,0.0001,0.001}, and weight decay∈ {0.0,0.001,0.01}. Final settings differed
between benchmarks (see Table 17).
Benchmark Hidden Dim Learning Rate
Benchmark I 64 0.001
Benchmark II 256 0.01
Table 17: Final FFNN hyperparameters. Other param-
eters were fixed at: dropout = 0.0, L1 = 0.001, weight
decay = 0.0.
Bidirectional LSTM (Bi-LSTM). Search space
included hidden size ∈ {64,128,256}and learning
rate∈ {0.01,0.001,0.0001}. The final configura-
tion used hidden size 64 and learning rate 0.0001
(see Table 18).
Benchmark Hidden Dim Learning Rate
Benchmark I 64 0.0001
Benchmark II 64 0.0001
Table 18: Final Bi-LSTM hyperparameters.
BERT Regression. We tuned dropout
∈ {0.0,0.2,0.4}and learning rate ∈ {0.0001}.
The best setting used no dropout and learning rate
0.0001. Training loss converged within 2 epochs
on both benchmarks (see Figure 2).
All scripts used for hyperparameter search and
training are available at: https://github.com/
krishgoel/chronocept-baseline-models .
16

# Annot ation Guidelines f or Chr onocept

This document pr o vides instructions f or annot ating t empor al v alidity using a * *thr ee-st ep pr ocess* *: * *T e xt Splitting* * ,  * *Axis A ssignment* * ,  
and * *T empor al V alidity Distribution Plotting* * .  
These guidelines ar e t ailor ed t o the natur e of this benchmark,  which typically inv olv es one * *M ain Axis* * segment and one additional axis 
segment fr om the se v en auxiliar y ax es.

## * *St ep 1: T e xt Splitting* *

### Objectiv e:
Divide the input sent ence int o gr ammatically corr ect segments,  ensuring semantic and t empor al int egrity is pr eser v ed.

### Guidelines:
1 .  * *Identify Splitting P oints:* *
   - Divide the sent ence int o meaningful subt e xts.  Most samples will include one * *M ain Axis* * segment and one fr om the other se v en ax es.
   - U se punctuation and conjunctions as natur al delimit ers but ensur e that each subt e xt is self -cont ained.
2.  * *Pr eser v e T empor al Cont e xt:* *
   - R et ain essential mark ers (e. g.,  *continuously* ,  *in 20 23* ,  *e v er y month*).
   - A v oid r emo ving or alt ering any t e xt.
3 .  * *A v oid Ov er -Splitting:* *
   - Ensur e each subt e xt conv e ys clear ,  st andalone meaning.
   - Ov er -splitting may lead t o fr agments that lose cont e xt or t empor al clarity .
4.  * *T e xt Cop ying Con v ention:* *
   - Cop y t e xt e x actly as it appears in the sample,  including punctuation.
5 .  * *Ex ample:* *
   - Input: * "The company is e xpanding its oper ations in A sia,  and the CEO is leading the eff or ts,  planning a significant incr ease in mark et shar e. " *
   - Split:
     - Subt e xt 1: "The company is e xpanding its oper ations in A sia, " (M ain Axis)
     - Subt e xt 2: "and the CEO is leading the eff or ts,  planning a significant incr ease in mark et shar e. " (Int ention Axis)
6 .  * *Ambiguity H andling:* *
   - If a sample seems t o violat e the condition of one M ain Axis plus one other axis,  document the * *Sample ID* * and consult * *Krish* * .
   - If a sample does not carr y a M ain Axis with a clearly definable t empor al cue,  document the * *Sample ID* * and consult * *Krish* * .
   - Incorr ect samples will be discar ded.

## * *St ep 2: Axis A ssignment* *

### Objectiv e:
Classify each subt e xt int o one of the * *se v en t empor al ax es* * based on its primar y t empor al char act eristic.
### T empor al Ax es:
1 .  * *M ain Axis (F actual Ev ents):* *
   - * *Definition* *: V erifiable e v ents along a timeline,  r epr esenting objectiv e truths.
   - * *Purpose* *: Captur es the primar y narr ativ e and est ablishes a concr et e t empor al sequence.
   - * *Ex ample* *: "The company is e xpanding its oper ations in A sia. "
   - * *K e y Question* *: Does this e v ent occur within the primar y timeline of the narr ativ e ?
2.  * *Int ention Axis:* *
   - * *Definition* *: Captur es someone's int ention,  desir e,  or plan,  e v en if unfulfilled.
   - * *Purpose* *: Highlights futur e-dir ect ed actions or goals tied t o the narr ativ e but not necessarily r ealiz ed.
   - * *Ex ample* *: "The CEO is leading the eff or ts,  planning a significant incr ease in mark et shar e. "
   - * *K e y Question* *: Is this e v ent st at ed as an int ended action or goal,  r egar dless of its r ealization?
3 .  * *Opinion Axis:* *
   - * *Definition* *: R epr esents subjectiv e viewpoints,  e xpect ations,  or beliefs about e v ents.
   - * *Purpose* *: Diff er entiat es opinions or speculations fr om f actual occurr ences.
   - * *Ex ample* *: "Exper ts belie v e the mark et will gr o w r apidly . "
   - * *K e y Question* *: Does this e v ent r epr esent a belief or e xpect ation r ather than a v erified f act?
4.  * *Hypothetical Axis:* *
   - * *Definition* *: Includes conditional or hypothetical e v ents dependent on cer t ain conditions.
   - * *Purpose* *: T r ack s scenarios that ar e imagined or conditional,  oft en using "if" st at ements.
   - * *Ex ample* *: "If the company secur es funding,  it will e xpand globally . "
   - * *K e y Question* *: Is this e v ent pr esent ed as dependent on another e v ent or condition?
5 .  * *Negation Axis:* *
   - * *Definition* *: Identifies e v ents e xplicitly st at ed as not occurring.
   - * *Purpose* *: T r ack s denied actions or out comes t o separ at e them fr om r ealiz ed e v ents.
   - * *Ex ample* *: "The company did not e xpand its oper ations in 20 20 . "
   - * *K e y Question* *: Is this e v ent e xplicitly st at ed as unfulfilled or negat ed?
6 .  * *Generic Axis:* *
   - * *Definition* *: R epr esents univ ersal truths or habitual occurr ences,  not tied t o a specific timeline.
   - * *Purpose* *: Highlights timeless f acts or gener alizations applicable br oadly .
   - * *Ex ample* *: "Lions eat meat. "
   - * *K e y Question* *: Is this e v ent a univ ersal truth or a habitual occurr ence that tr anscends specific cont e xts?
7 .  * *St atic Axis:* *
   - * *Definition* *: Captur es unchanging st at es or conditions * *within a specific cont e xt or timefr ame* * .
   - * *Purpose* *: T r ack s cont e xt -dependent f acts or conditions r ele v ant t o the narr ativ e.
   - * *Ex ample* *: "The r oom is cold. "
   - * *K e y Question* *: Is this e v ent cont e xt -specific and st atic within the described situation?
8 .  * *R ecurr ent Axis:* *
   - * *Definition* *: Describes e v ents or st at es that happen r epeat edly o v er time.
   - * *Purpose* *: T r ack s patt erns or cy cles of actions/ e v ents r ele v ant t o the narr ativ e.
   - * *Ex ample* *: "The tr ain arriv es e v er y morning at 8 AM. "
   - * *K e y Question* *: Does this e v ent r epr esent a r ecurring action or patt ern?

### Guidelines:
1 .  * *A ssign t o the Closest Axis:* *
   - Car efully analyz e the t empor al and semantic meaning of the subt e xt.
   - Decide if the e v ent can be anchor ed t o a specific axis based on its natur e.
   - Most samples will hav e one * *M ain Axis* * subt e xt and one auxiliar y axis subt e xt.
2.  * *H andle Ambiguities:* *
   - F ocus on the st ar t -points of e v ents t o r educe ambiguity r elat ed t o dur ations.
   - Only compar e e v ents on the same axis; cr oss-axis r elations r equir e separ at e inv estigation.
   - If unsur e about the axis,  document the * *Sample ID* * and consult * *Krish* * .
   - Incorr ect samples will be r emo v ed fr om the dat aset.
3 .  * *U se Cont e xt:* *
   - A ssess the br oader cont e xt t o distinguish betw een ax es lik e St atic and Generic.
4.  * *Ex ample Annot ation:* *
   - Subt e xt: "The CEO is leading the eff or ts,  planning a significant incr ease in mark et shar e. "
     - A ssigned Axis: * *Int ention Axis* *
5 .  * *A dvisor y f or Comple x C ases:* *
   - Consider the f ollo wing e x ample: "The print er is making str ange noises while the IT t echnician tries t o fix it. "
     - "The IT t echnician is tr ying t o fix the print er" can be tr eat ed as the * *M ain Axis* * ,  while "the print er is making str ange noises" can be assigned 
t o the * *Generic Axis* * .
     - This r equir es thoughtful analysis,  as the r oles of subt e xts may not be appar ent immediat ely .  Annot at ors should car efully consider such cases,  
akin t o tr ansposing the segments f or clarity .

## * *St ep 3: T empor al V alidity Distribution Plotting* *

### Objectiv e:
Plot a sk ew ed pr obability distribution o v er a * *time gr aph* * t o r epr esent the t empor al v alidity of each subt e xt.

### Guidelines:
1 .  * *T empor al Cue A ssignment:* *
   - F or samples with clear t empor al cues (e. g.,  "solving f or 1 hour"),  assign a time int er v al t o that cue.  A s an advisor y ,  consider that a v ernacularly 
used "1 hour" can r ange fr om 45 minut es t o 90 minut es.
2.  * *Gr aph Ax es:* *
   - * *X - Axis (T ime):* *
     - Labeled with int er v als: 1 minut e,  15 minut es,  30 minut es,  1 hour ,  12 hours,  1 day ,  1 w eek,  1 month,  1 y ear ,  1 decade,  and infinit e v alidity .
   - * *Y - Axis (Pr obability):* *
     - R ange: 0 (not v alid) t o 1 (fully v alid).
3 .  * *Plotting P oints:* *
   - Place 3–5 points on the timeline t o indicat e the pr obability of v alidity at specific times.
   - The user need not w orr y about making an ideal pr obability distribution with * *A UC = 1* * .  Inst ead,  plot pr opor tions r elativ e t o the t empor al 
"point" with the highest pr obability (M aximum Lik elihood Estimat e,  MLE).
4.  * *Fit a Sk e w ed Pr obability Distribution:* *
   - A sk ew ed cur v e will be aut omatically fitt ed thr ough the plott ed points t o r epr esent the t empor al v alidity distribution.
5 .  * *Consist ency:* *
   - M aint ain consist ency in plotting f or similar subt e xts.
6 .  * *Ambiguity H andling:* *
   - If the sample is t echnically corr ect but y ou ar e highly unsur e about the t empor al int er v al,  annot at e t o the best of y our ability .  L o w int er -annot at or 
agr eement (IAA) samples will be flagged and eliminat ed 
during post -pr ocessing.
   - If unsur e about the distribution,  document the * *Sample ID* * and consult * *Krish* * .
   - Incorr ect samples will be r emo v ed.
The r esult of this st ep is a sk ew ed pr obability distribution r eflecting the t empor al v alidity o v er time.

## * *Gener al Not es f or Annot at ors* *

1 .  * *Ambiguities:* *
   - F or unclear splits,  axis assignments,  or v alidity distributions,  cont act * *Krish* * with the * *Sample ID* * f or r esolution.
2.  * *Discar ding Samples:* *
   - Multimodal samples or those with e x cessiv e ambiguity should be flagged f or r e view and pot ential r emo v al.
3 .  * *T empor al Objectivity:* *
   - A v oid consulting peers during annot ation t o maint ain objectivity and ensur e consist ency acr oss annot at ors.
4.  * *Quality Contr ol:* *
   - Ensur e all annot ations ar e thor ough,  consist ent,  and adher e t o these guidelines.Figure 3: Annotation guidelines for Chronocept.
17

15 min 30 min
time [t' = log1.1(t)]0.00.10.20.30.40.50.60.70.8P(validity)Scenario: Early Onset
'He is making coffee for himself right now.'
Data Points
Skew-Normal
Gaussian
Exponential
Log-Normal
Gamma(a) Early Onset: Peak validity occurs soon after publication.
time [t' = log1.1(t)]0.10.20.30.40.50.60.70.8P(validity)Scenario: Late Onset
'The movie is going to hit the theaters in a few weeks.'
Data Points
Skew-Normal
Gaussian
Exponential
Log-Normal
Gamma (b) Late Onset: Validity emerges gradually and peaks later.
15 min 30 min
time [t' = log1.1(t)]0.10.20.30.40.50.60.70.8P(validity)Scenario: Short Duration
'The site has been crashing for a few minutes as there is
some server maintenance work going on.'
Data Points
Skew-Normal
Gaussian
Exponential
Log-Normal
Gamma
(c) Short Duration: A narrow window of high relevance.
1 min15 min 30 min1 hr
time [t' = log1.1(t)]0.00.20.40.60.8P(validity)Scenario: Long Duration
'The ruling government brings growth and progress.'
Data Points
Skew-Normal
Gaussian
Exponential
Log-Normal
Gamma (d) Long Duration: Validity persists over time.
1 hr
time [t' = log1.1(t)]0.00.20.40.60.8P(validity)Scenario: Rapid Rise, Slow Decay
'The advertisement s impact peaks immediately and lingers.'
Data Points
Skew-Normal
Gaussian
Exponential
Log-Normal
Gamma
(e) Rapid Rise, Slow Decay: Sudden onset, gradual decline.
1 hr
time [t' = log1.1(t)]0.00.20.40.60.81.0P(validity)Scenario: Slow Rise, Rapid Decay
'The news slowly gains attention but quickly becomes outdated.'
Data Points
Skew-Normal
Gaussian
Exponential
Log-Normal
Gamma (f) Slow Rise, Rapid Decay: Gradual onset, sharp drop.
Figure 6: Visual fit comparison of candidate distributions across six temporal scenarios. The skew-normal
consistently provides the best fit, modeling varied validity patterns in onset, duration, and asymmetry.
18

#  S yn th e t ic  D a t a  G ene r a t ion  f o r a  T empo r a l V a l idi t y  Benc h ma r k 
##  Objec t iv e 
Thi s t as k  inv olv e s cr e a ting  s ynthetic  s ent ence s th a t  will  f orm  the  b as i s of  a benchm a rk  f or  t empor a l  v a lidity  r e s e a r ch.  Y our  
r ole  as a t e xt  gener a tion  model  i s t o  pr oduce  *high-qu a lity  s ent ence s only* ,  without  a ccomp a nying  e xpl a n a tion s or  a xi s 
de s cription s .  The s e  s ent ence s s hould  de s cribe  occurr ence s or  e v ent s th a t  h a ppen  s imult a neou s ly  or  contr as tiv ely ,  
incorpor a ting  v a riou s a ction s ,  s t a t e s ,  or  pr oce ss e s . 
##  K e y  Defini t ion:  Axis 
An  a xi s r epr e s ent s a s em a ntic  dimen s ion  or  ch a r a ct eri s tic  u s ed  t o  cl ass ify  a nd  a n a lyz e  the  r el a tion s hip s betw een  e v ent s in  
a s ent ence.  Ax e s a r e  c a t egoriz ed  int o  tw o  type s : 
1 .  * *Ev en t -R e l a t ed  Ax es* *:  De s cribe  the  r el a tion s hip  betw een  e v ent s or  s t a t e s in  a s ent ence,  f ocu s ing  on  int er a ction s or  
dependencie s .    
2.  * *Anno t a t ion  Ax es* *:  Pr o vide  s upplement a ry  s em a ntic  inf orm a tion  a bout  the  e v ent s ,  enh a ncing  int erpr et a bility .    
### Ev en t -R e l a t ed Ax es 
Specify  the  r el a tion s hip  betw een  e v ent s in  the  s ent ence:    
1 .  * *T empo r a l Ov e rl ap* *:  Ev ent s occur  s imult a neou s ly  or  in  p a r a llel.    
2.  * *C ausa l i t y* *:  One  e v ent  c a u s e s or  r e s ult s fr om  the  other .    
3 .  * * S ubo r dina t ion* *:  One  e v ent  depend s on  or  occur s due  t o  the  other .    
4.  * *Un r e l a t ed* *:  Ev ent s a r e  independent  of  e a ch  other .    
### Anno t a t ion Ax es 
Pr o vide  s em a ntic  cont e xt  a nd  a ddition a l  dimen s ion s of  me a ning:    
1 .  * *M ain  Axis  (F ac t ua l Ev en t s)**:  V erifi a ble,  objectiv e  e v ent s tied  t o  a s pecific  timeline.    
2.  * *In t en t ion* *:  Futur e-dir ect ed  pl a n s ,  de s ir e s ,  or  a ction s .    
3 .  * *Opinion* *:  Subjectiv e  belief s or  e xpect a tion s a bout  e v ent s .    
4.  * *Hypo th e t ica l * *:  Condition a l  or  im a gined  s cen a rio s .    
5 .  * *Nega t ion* *:  Explicitly  unfulfilled  or  denied  a ction s or  out come s .    
6 .  * * G ene r ic* *:  Univ er sa l  truth s or  h a bitu a l  a ction s th a t  a pply  br o a dly  a cr o ss cont e xt s a nd  a r e  not  tied  t o  s pecific  timeline s .    
7 .  * * St a t ic* *:  Unch a nging  s t a t e s or  condition s th a t  a r e  s pecific  t o  a p a rticul a r  cont e xt  or  timefr a me.    
8 .  * *R ecu rr en t * *:  Ev ent s or  s t a t e s th a t  r ecur  o v er  time,  f orming  p a tt ern s or  cy cle s .    
##  G uide l ines  f o r S en t ence  G ene r a t ion 
### S en t ence Str uc t u r e 
-  Sent ence s s hould  be  writt en  in  the  *pr e s ent t en s e* .  U s e  * *a ll  f o r ms of p r esen t  t ense* *  -  Simple  Pr e s ent  T en s e,  Pr e s ent  
Continuou s T en s e,  Pr e s ent  P erf ect  T en s e  a nd  Pr e s ent  P erf ect  Continuou s T en s e. 
-  E a ch  s ent ence  s hould  incorpor a t e:    
  -  * A t le as t one Ev ent -R el a t ed Axi s *  t o  define  the  r el a tion s hip  betw een  e v ent s .    
  -  * T w o Annot a tion Ax e s ,  one of which mu s t be the * *M ain Axis (F ac t ua l  Ev en t s)* * * .    
## Neu tr a l i t y and Div e r si t y 
-  Sent ence s mu s t  s p a n  *div er s e dom a in s * ,  including  d a ily  lif e,  t echnology ,  a b s tr a ct  concept s ,  a nd  n a tur e.    
-  U s e  a mix  of  *pr onoun s *  ("he, "  " s he, "  "the y "),  *generic entitie s *  (e. g.,  " a per s on, "  " a m a chine"),  a nd  * a rticle s *  ("the, "  " a ").  
En s ur e  pr onoun s a r e  e v enly  di s tribut ed  a cr o ss the  d a t as et  t o  r epr e s ent  div er s e  a ct or s . 
##  T ask  Ou t pu t 
1 .  Gener a t e  *50 s ent ence s *  a dhering  s trictly  t o  the  a bo v e  s tructur e  a nd  r equir ement s .    
2.  En s ur e  div er s ity  in  dom a in s ,  a x e s ,  a nd  e v ent  r el a tion s hip s while  m a int a ining  cl a rity  a nd  coher ence.    
3 .  E a ch  s ent ence  mu s t  e xplicitly  include:    
   -  * *A t  l eas t  one Ev en t -R e l a t ed Axis* * . 
   -  * *T w o Anno t a t ion Ax es* * ,  with  the  *M a in Axi s  (F a ctu a l Ev ent s )*  included.    
##  Ex amp l es  of  Co rr ec t S en t ences 
1 .  "She  i s cooking  dinner ,  but  the  o v en  k eep s m a lfunctioning. "    
2.  "He  i s driving  t o  w ork,  while  the  tr a ffic  j a m  i s w or s ening. "    
3 .  "The y  a r e  r e viewing  document s ,  as the  de a dline  a ppr o a che s . "    
4.  " A  r e s e a r cher  i s de s igning  a n  e xperiment,  while  the  t echnici a n  pr ep a r e s the  equipment. "    
5 .  "The  s ky  i s d a rk ening,  but  the  l a k e  r em a in s c a lm  a nd  s till. "    
6 .  " A  s tudent  i s r e a ding  the  m a nu a l  t o  under s t a nd  ho w  the  de vice  might  oper a t e. "    
7 .  "She  i s negoti a ting  a contr a ct,  while  her  t e a m  fin a liz e s the  pr e s ent a tion. "    
8 .  "The  cloud s a r e  g a thering,  a nd  the  wind  i s picking  up  s peed. "    
9 .  "The  r obot  i s perf orming  a t as k,  while  the  oper a t or  monit or s it s efficiency . "    
10 .  "He  i s pr a cticing  the  pi a no ,  but  the  a udience  r em a in s   s ilent. "Figure 7: Plaintext markdown prompt for Benchmark I.
19

#  S yn th e t ic  D a t a  G ene r a t ion  f o r a  T em p o r a l V a l idi t y  Benc h ma r k 
##  O b jec t iv e 
Y our  r ole  as a t e xt  ge n er a tio n model  i s t o  pr oduce  *high - qu a lity ,  coher e n t,  an d  na tur a lly  flo wi n g  s e n t e n ce s or  s hort  
p a r a gr a ph s * ,  without  a ccomp an yi n g  e xpl ana tio ns or  a xi s de s criptio ns .  The s e  sa mple s s hould  de s cribe  occurr e n ce s or  
e v e n t s th a t  h a ppe n s imult an eou s ly  or  co n tr as ti v ely ,  i n corpor a ti n g  v a riou s a ctio ns ,  s t a t e s ,  or  pr oce ss e s .  A v oid  u nna tur a l,  
o v erly  f orm a l,  or  s tilt ed  co ns tructio ns . 
##  K e y  De f ini t ion:  Axis 
A n a xi s r epr e s e n t s a s em an tic  dime ns io n or  ch a r a ct eri s tic  u s ed  t o  cl ass ify  an d  ana ly z e  the  r el a tio ns hip s betw ee n e v e n t s i n 
a s e n t e n ce.  Ax e s a r e  c a t egori z ed  i n t o  tw o  type s : 
1 .  * *Ev en t -R e l a t ed  Ax es* *:  D e s cribe  the  r el a tio ns hip  betw ee n e v e n t s or  s t a t e s i n a s e n t e n ce,  f ocu s i n g  o n i n t er a ctio ns or  
depe n de n cie s .    
2.  * *Anno t a t ion  Ax es* *:  Pr o v ide  s uppleme n t a ry  s em an tic  i n f orm a tio n a bout  the  e v e n t s ,  e n h an ci n g  i n t erpr et a bility .    
###  Ev en t -R e l a t ed  Ax es 
Specify  the  r el a tio ns hip  betw ee n e v e n t s i n the  s e n t e n ce:    
1 .  * *T em p o r a l Ov e rl a p * *:  E v e n t s occur  s imult an eou s ly  or  i n p a r a llel.    
2.  * *C ausa l i t y* *:  O n e  e v e n t  c a u s e s or  r e s ult s fr om  the  other .    
3 .  * * S u b o r dina t ion* *:  O n e  e v e n t  depe n d s o n or  occur s due  t o  the  other .    
4.  * *Un r e l a t ed* *:  E v e n t s a r e  i n depe n de n t  of  e a ch  other .    
###  Anno t a t ion  Ax es 
Pr o v ide  s em an tic  co n t e xt  an d  a dditio na l  dime ns io ns of  me an i n g:    
1 .  * *M ain  Axis  (F ac t ua l Ev en t s)**:  V erifi a ble,  objecti v e  e v e n t s tied  t o  a s pecific  timeli n e.    
2.  * *In t en t ion* *:  Futur e - dir ect ed  pl ans ,  de s ir e s ,  or  a ctio ns .    
3 .  * *O p inion* *:  Subjecti v e  belief s or  e xpect a tio ns a bout  e v e n t s .    
4.  * *Hy p o th e t ica l * *:  Co n ditio na l  or  im a gi n ed  s ce na rio s .    
5 .  * *Nega t ion* *:  Explicitly  u n fulfilled  or  de n ied  a ctio ns or  out come s .    
6 .  * * G ene r ic* *:  U n i v er sa l  truth s or  h a bitu a l  a ctio ns th a t  a pply  br o a dly  a cr o ss co n t e xt s an d  a r e  n ot  tied  t o  s pecific  timeli n e s .    
7 .  * * St a t ic* *:  U n ch an gi n g  s t a t e s or  co n ditio ns th a t  a r e  s pecific  t o  a p a rticul a r  co n t e xt  or  timefr a me.    
8 .  * *R ecu rr en t * *:  E v e n t s or  s t a t e s th a t  r ecur  o v er  time,  f ormi n g  p a tt er ns or  cy cle s .    
##  G uide l ines  f o r S en t ence  G ene r a t ion 
###  S en t ence  Str uc t u r e 
- Se n t e n ce s s hould  be  writt e n i n the  *pr e s e n t t e ns e* .  U s e  * *a ll  f o r ms o f  pr esen t  t ense* *  - Simple  Pr e s e n t  T e ns e,  Pr e s e n t  
Co n ti n uou s T e ns e,  Pr e s e n t  P erf ect  T e ns e  an d  Pr e s e n t  P erf ect  Co n ti n uou s T e ns e. 
- E a ch  s e n t e n ce  s hould  i n corpor a t e:    
  - * A t le as t tw o E v e n t - R el a t ed Ax e s *  t o  defi n e  the  r el a tio ns hip  betw ee n e v e n t s .    
  - *F our or mor e A nn ot a tio n  Ax e s * ,  o n e of which mu s t be the * *M ain Axis (F ac t ua l  Ev en t s)* * .  
- A v oid  o v eru s i n g  comm as .  Ins t e a d,  u s e  full  s t op s t o  s ep a r a t e  ide as i n t o  di s ti n ct  s e n t e n ce s wher e  a ppr opri a t e. 
## Neu tr a l i t y and Div e r si t y 
- Se n t e n ce s mu s t  s p an *di v er s e dom a i ns * ,  i n cludi n g  d a ily  lif e,  t ech n ology ,  a b s tr a ct  co n cept s ,  an d  na tur e.    
- U s e  a mix  of  *pr o n ou ns *  ("he, "  " s he, "  "the y "),  *ge n eric e n titie s *  (e. g.,  " a per s o n , "  " a m a chi n e"),  an d  * a rticle s *  ("the, "  " a ").  
E ns ur e  pr o n ou ns a r e  e v e n ly  di s tribut ed  a cr o ss the  d a t as et  t o  r epr e s e n t  di v er s e  a ct or s . 
##  T ask  Ou tp u t 
1 .  Ge n er a t e  *50 s e n t e n ce s *  a dheri n g  s trictly  t o  the  a bo v e  s tructur e  an d  r equir eme n t s .    
2.  E ns ur e  di v er s ity  i n dom a i ns ,  a x e s ,  an d  e v e n t  r el a tio ns hip s while  m a i n t a i n i n g  cl a rity  an d  coher e n ce.    
3 .  E a ch  s e n t e n ce  mu s t  e xplicitly  i n clude:    
   - * *A t  l eas t  t w o Ev en t -R e l a t ed Axis* * . 
   - * *F ou r  o r  mo r e Anno t a t ion Ax es* * ,  with  the  *M a i n  Axi s  (F a ctu a l E v e n t s )*  i n cluded.    
##  Ex am pl es  o f Co rr ec t S en t ences 
1 .  “ She  i s cooki n g  di nn er .  A t  the  sa me  time,  the  o v e n i s m a lfu n ctio n i n g,  which  c a u s e s del a y s i n her  pr ep a r a tio n .  She  check s 
the  i n gr edie n t s r epe a t edly ,  e ns uri n g  n othi n g  i s mi ss i n g,  while  w orryi n g  th a t  the  di s h  m a y  n ot  tur n out  as pl ann ed.  D e s pit e  
the  ch a lle n ge s ,  s he  i n t e n d s t o  s er v e  the  me a l  o n time  t o  s urpri s e  her  f a mily . ”
2.  “He  i s dri v i n g  t o  w ork,  nav ig a ti n g  thr ough  de ns e  tr a ffic  as the  mor n i n g  ru s h  i n t e ns ifie s .  Me an while,  the  tr a ffic  j a m  
w or s e ns due  t o  a n e a rb y  a ccide n t,  f or ci n g  him  t o  r ethi n k  hi s r out e  while  c a lcul a ti n g  the  e s tim a t ed  del a y .  He  co ns ider s 
t a ki n g  a det our  thr ough  s ide  s tr eet s ,  hopi n g  t o  sav e  time,  but  w orrie s it  might  le a d  t o  further  complic a tio ns . ”
3 .  “ She  i s w a t eri n g  the  g a r de n while  the  s u n r em a i ns hidde n behi n d  the  cloud s ,  le a di n g  t o  s lo w er  e v a por a tio n .  She  
fr eque n tly  check s the  s oil  moi s tur e,  belie v i n g  th a t  o v erw a t eri n g  might  d a m a ge  the  pl an t s ,  though  s he  i n t e n d s t o  u s e  
or g an ic  f ertili z er  s oo n . ”Figure 8: Plaintext markdown prompt for Benchmark II.
20