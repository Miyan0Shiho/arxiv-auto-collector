# Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection

**Authors**: Xuwei Tan, Yao Ma, Xueru Zhang

**Published**: 2025-12-15 07:09:11

**PDF URL**: [https://arxiv.org/pdf/2512.13040v1](https://arxiv.org/pdf/2512.13040v1)

## Abstract
Detecting fraud in financial transactions typically relies on tabular models that demand heavy feature engineering to handle high-dimensional data and offer limited interpretability, making it difficult for humans to understand predictions. Large Language Models (LLMs), in contrast, can produce human-readable explanations and facilitate feature analysis, potentially reducing the manual workload of fraud analysts and informing system refinements. However, they perform poorly when applied directly to tabular fraud detection due to the difficulty of reasoning over many features, the extreme class imbalance, and the absence of contextual information. To bridge this gap, we introduce FinFRE-RAG, a two-stage approach that applies importance-guided feature reduction to serialize a compact subset of numeric/categorical attributes into natural language and performs retrieval-augmented in-context learning over label-aware, instance-level exemplars. Across four public fraud datasets and three families of open-weight LLMs, FinFRE-RAG substantially improves F1/MCC over direct prompting and is competitive with strong tabular baselines in several settings. Although these LLMs still lag behind specialized classifiers, they narrow the performance gap and provide interpretable rationales, highlighting their value as assistive tools in fraud analysis.

## Full Text


<!-- PDF content starts -->

Understanding Structured Financial Data with LLMs:
A Case Study on Fraud Detection
Xuwei Tan*
The Ohio State University
Coinbase, Inc.Yao Ma
Coinbase, Inc.Xueru Zhang
The Ohio State University
Abstract
Detecting fraud in financial transactions typ-
ically relies on tabular models that demand
heavy feature engineering to handle high-
dimensional data and offer limited interpretabil-
ity, making it difficult for humans to understand
predictions. Large Language Models (LLMs),
in contrast, can produce human-readable expla-
nations and facilitate feature analysis, poten-
tially reducing the manual workload of fraud
analysts and informing system refinements.
However, they perform poorly when applied
directly to tabular fraud detection due to the
difficulty of reasoning over many features, the
extreme class imbalance, and the absence of
contextual information. To bridge this gap, we
introduce FinFRE-RAG, a two-stage approach
that applies importance-guided feature reduc-
tion to serialize a compact subset of numer-
ic/categorical attributes into natural language
and performs retrieval-augmented in-context
learning over label-aware, instance-level exem-
plars. Across four public fraud datasets and
three families of open-weight LLMs, FinFRE-
RAG substantially improves F1/MCC over di-
rect prompting and is competitive with strong
tabular baselines in several settings. Although
these LLMs still lag behind specialized clas-
sifiers, they narrow the performance gap and
provide interpretable rationales, highlighting
their value as assistive tools in fraud analysis.
1 Introduction
Financial fraud is a pervasive and evolving threat
that costs businesses and consumers billions of
dollars every year (Hilal et al., 2022). The fraud
landscape is highly dynamic, with malicious actors
continually devising new strategies to circumvent
existing safeguards. In response to the limitations
of rule-based systems (Soui et al., 2019), machine
*Work done during internship at Coinbase. This paper con-
tains the author’s personal opinions and does not constitute a
company policy or statement. These opinions are not endorsed
by or affiliated with Coinbase, Inc. or its subsidiaries.learning (ML) models have been adopted to iden-
tify illicit or abnormal transactions from large-scale
financial data (Jin and Zhang, 2025). However,
their performance is critically dependent on ex-
tensive and laborious feature engineering. This
reliance makes them costly to develop and main-
tain. Furthermore, these models are sensitive to
pattern shifts (Gama et al., 2014), where the statis-
tical properties of the data change over time, requir-
ing frequent retraining and model updates. These
trained models also offer limited interpretability be-
yond aggregate feature importance scores, posing
challenges to transparent decision-making.
Recent advances in LLMs offer a promising al-
ternative across many domains (Wang et al., 2025;
Hu et al., 2024; Ning et al., 2025; Wang et al.,
2024), including finance (Wu et al., 2023; Wang
et al.; Yu et al., 2024). With strong capabilities
in complex reasoning and emulating the logic of
human analysts (Brown et al., 2020), LLMs hold
the potential to not only detect fraud but also pro-
vide human-readable rationales for their decisions.
However, prior studies typically use LLMs as aux-
iliary tools to augment traditional classifiers rather
than as standalone detectors. For example, (Yang
et al., 2025b) and (Huang and Wang, 2025) inte-
grate LLMs to improve the performance of Graph
Neural Networks in fraud detection. Nevertheless,
applying LLMs directly to fraud detection remains
challenging. Benchmarks (Feng et al., 2023; Xie
et al., 2024) highlight that existing LLMs still strug-
gle with financial risk prediction, achieving only
marginal improvements over random guessing.
A key question arises: How can we adapt LLMs
to understand and detect fraud in tabular transaction
data? This challenge stems from a core misalign-
ment between the LLMs and the domain realities.
Specifically, we have identified two key issues:
•Tabular Input Misalignment:Unlike frauds
such as internet scams or phone scams that in-
volve rich text data (Yang et al., 2025c; Singh
1arXiv:2512.13040v1  [cs.LG]  15 Dec 2025

et al., 2025), financial transaction data are pri-
marily numeric and categorical features in a tab-
ular format. LLMs, being pretrained mainly on
natural language, may not interpret the semantic
meaning of structured features or handle numer-
ical values with the required precision. In addi-
tion, real transactions can also involve hundreds
or thousands of features, including all of them
in a prompt may exceed context limits and intro-
duce noise, as many are less significant signals.
This lack of semantic context and high dimen-
sionality makes it difficult for LLMs to detect
fraud patterns from raw tabular data.
•Fraud Ambiguity and Rarity:Fraudulent trans-
actions typically require nuanced, expert knowl-
edge to recognize, and what constitutes “fraud”
can vary greatly across different institutions,
product types, or regions. An LLM without
domain-specific guidance can be easily con-
founded by this variability in fraud definitions
and patterns, leading to unpredictable or nonsen-
sical outputs. Additionally, fraud datasets are
usually extremely imbalanced, where the pro-
portion of fraudulent instances may be less than
1% of all transactions. LLMs may struggle to
identify the subtle patterns that differentiate a
legitimate transaction from a fraudulent one.
The key is to teach LLMs what constitutes fraud
and which feature patterns often lead to fraudulent
behavior. Without explicit guidance, LLMs tend to
treat tabular inputs as arbitrary numbers, producing
nearly random predictions. To narrow this gap, we
leverage selective historical transactions as few-
shot exemplars, providing the model with concrete
cases of both fraudulent and legitimate behavior.
By grounding the task in real examples, the LLM
gains a contextual understanding of how subtle
feature patterns correspond to risk.
To this end, this paper introduces aFINancial
Feature-REduced in-context learning framework
withRetrieval-AugmentedGeneration (FinFRE-
RAG), designed to harness the reasoning capabili-
ties of LLMs while addressing their inherent weak-
nesses in structured fraud detection. FinFRE-RAG
is built on two key stages: the offline feature re-
duction for prompt construction with the most im-
pactful features, ensuring compact yet informative
representations; and the online retrieval-augmented
in-context learning to construct a temporary, task-
relevant micro-dataset within the LLM’s context
window. Our method recasts fraud detection as
an instance-based reasoning problem. By supply-ing few-shot examples, the LLM adapts to each
query without the computational overhead of train-
ing. Our contributions are summarized as follows:
•We proposeFinFRE-RAG, a two-stage frame-
work that adapts LLMs to structured fraud detec-
tion without task-specific training.
•We conduct a systematic evaluation across mul-
tiple fraud benchmarks and open-weight LLMs,
showing substantial F1/MCC gains bridging prior
gaps by demonstrating that, with targeted feature
reduction and retrieval, LLMs no longer system-
atically lag behind traditional methodologies.
•We provide principled analyses that clarify how
to use LLMs for tabular fraud by quantifying the
required number of features and similar transac-
tions, assessing the impact of prediction granu-
larity, and comparing against fine-tuned LLMs.
2 Related Work
2.1 Fraud Detection
Traditional approaches for fraud detection relied
heavily on rule-based systems and hand-crafted
features, which, while interpretable, often strug-
gled to adapt to evolving fraud patterns. In recent
years, with the increasing availability of large-scale
transaction data and the advances in deep learn-
ing, research has shifted towards more expressive
models that capture more complex patterns. Tradi-
tional machine learning methods, particularly en-
semble algorithms such as XGBoost (Chen and
Guestrin, 2016), LightGBM (Ke et al., 2017), and
CatBoost (Prokhorenkova et al., 2018), have shown
strong effectiveness in capturing non-linear inter-
actions among structured features. Beyond ensem-
bles, deep neural networks have also been applied
to detect different kinds of fraud (Fiore et al., 2019;
Dou et al., 2020; Li et al., 2024; Yu et al., 2023).
2.2 LLMs for finance
A growing line of research has emphasized the
importance of benchmarking LLMs for financial
applications. Xie et al. (2023, 2024) build a large-
scale benchmark covering a wide range of financial
tasks, including fraud detection. InvestorBench
(Li et al., 2025) evaluates LLM-based agents in
financial decision-making settings, such as stock
and cryptocurrency trading. Beyond text, FinMME
(Luo et al., 2025) and FCMR (Kim et al., 2025)
extended the scope to multi-modal evaluation, in-
corporating tables, charts, and textual reports.
Parallel to benchmark construction, researchers
2

Transaction 
Request
Transaction 
FeaturesHistoric 
Transactions Feature
 Importance Ranking
Selected Top k
FeaturesFeature
Normalization
Vector 
DatasetRetriever
Similar
Transactions Feature 
Building
QueryLLMRisk Score : 4
Explanation :Block / Allow
Human 
ReviewerThe current case aligns 
closely with the majority 
of prior fraud cases for 
state 5, similar credit 
balance  ...Given the 
strong similarity to the 
fraud group, the 
probability is highTransaction Prompt
Current case: The client state number is 5, the 
number of cards is 1, the credit balance is 10540 , 
the number of transactions is 61, the number of 
international transactions is 5, the credit limit is 25.Transaction Prompt
Current case: The client state number is 5, the 
number of cards is 1, the credit balance is 10540 , 
the number of transactions is 61, the number of 
international transactions is 5, the credit limit is 25.
Retrieved Transactions
You are given several similar historical cases with 
their ground truth labels. Use them as guidance 
and then assess the current case. 
Example 1: ……: It is a fraud.Retrieved Transactions
You are given several similar historical cases with 
their ground truth labels. Use them as guidance 
and then assess the current case. 
Example 1: ……: It is a fraud.System Prompt
You are a helpful financial expert that can help 
analyze fraud .... Please give a range of 1 to 5 for 
the probability of fraud.  And please give a brief 
explanation for your score.System Prompt
You are a helpful financial expert that can help 
analyze fraud .... Please give a range of 1 to 5 for 
the probability of fraud.  And please give a brief 
explanation for your score.
Online OfflineTop nFigure 1: Overall architecture of the FinFRE-RAG framework.
have also introduced a series of domain-specific
financial LLMs. Early efforts include FinBERT
(Yang et al., 2020), which adapted BERT for finan-
cial sentiment analysis; BloombergGPT (Wu et al.,
2023), a large-scale model trained on proprietary
financial corpora; and FinGPT (Wang et al.), which
emphasizes open, low-cost, and continuously up-
dated pipelines. Beyond general-purpose models,
several frameworks have been designed for task-
specific applications. For example, Yu et al. (2024)
proposed a multi-agent framework for stock trad-
ing and portfolio management, while Aguda et al.
(2024) explored leveraging LLMs as data annota-
tors to extract relations in financial documents.
2.3 In-context learning and RAG
LLMs can adapt to new tasks at inference by con-
ditioning on a few input–label examples in the
prompt, without parameter updates (Brown et al.,
2020).Thisin-context learning(ICL) enables few-
shot generalization through text alone. Prior work
suggests that ICL may emulate implicit learning
mechanisms, such as pattern matching or regres-
sion, while remaining sensitive to the choice and
order of demonstrations (Min et al., 2022).
Retrieval-augmented generation (RAG) equips
LLMs with external memory: a retriever selects
relevant examples, and the generator conditions on
them to produce grounded outputs (Lewis et al.,
2020). The integration of external knowledge
can enhance the accuracy of model responses and
improve generation reliability (Guu et al., 2020;
Karpukhin et al., 2020; Ram et al., 2023). We
extend this idea to structured data: retrieve seman-
tically similar instances, present them as context,
and let the model reason by comparison.3 Proposed Method
In this section, we introduce FinFRE-RAG; the
overall workflow is illustrated in Figure 1. It con-
sists of two key stages: offline feature reduction
and online retrieval-augmented in-context learning.
3.1 Problem Setup
LetFdenote the set of available features with
dimensionality d=|F| . We consider two
datasets: (1)External historical dataset: Dext=
{(xi, yi)}N
i=1. Eachx i∈Rdis a transaction repre-
sented by the full feature set F. The corresponding
label yi∈ {0,1} indicates whether the transac-
tion is fraudulent ( yi= 1) or legitimate. (2)Test
dataset: Dtest={x q}M
q=1. It contains Munla-
beled online query transactions, eachx q∈Rd
Our goal is to design a prompting strategy that
enables an LLM, denoted by hθ(·), to classify a
query transaction xq∈ D test. The LLM should
achieve this by retrieving relevant examples from
Dextand providing a risk score with natural lan-
guage explanations for its decision, without fine-
tuning of its parametersθ.
3.2 Offline Feature Reduction
Real-world transaction data often includes hun-
dreds or even thousands of features, many of which
carry less significant signal for fraud detection. If
such high-dimensional data are fed directly into
a retrieval-augmented LLM (Lewis et al., 2020),
two problems arise: (i) long prompts exceed the
model’s context window, making it infeasible to
include all features and examples due to extra infer-
ence time and memory requirements, and (ii) irrel-
evant or noisy features dilute the predictive signal
and hinder the model’s ability to learn meaningful
patterns. Thus, in FinFRE-RAG, we consider re-
ducing the feature space to ensure that the LLM
3

focuses on the most informative attributes and that
retrieval remains efficient.
To address this, we perform an offline feature
selection step that leverages the strengths of tradi-
tional machine learning models, which are used to
capture the initial predictive patterns in structured
data. Specifically, we train a Random Forest on
the external dataset Dextand extract feature im-
portance scores to rank all dfeatures in F. The
goal is not to build a production-grade model, but
to obtain a rough yet effective ranking at minimal
cost, avoiding the expense of hyperparameter op-
timization. From this ranking, we retain the top- k
features ( k < d ), yielding a reduced feature set
Fsel⊂ F. This reduction ensures that subsequent
retrieval-augmented prompts concentrate on the
most relevant attributes, allowing the LLM to oper-
ate within its context limits.
To further align tabular structure with retrieval
efficiency, we precompute standardized representa-
tions for all numeric features in Fsel. Specifically,
for each feature f∈ F sel, we calculate global statis-
tics(µ f, σf)onD extand normalize as:
zf(x) =x[f]−µ f
σf.(1)
which yields a vectorized dataset where every trans-
action is embedded in a comparable space. By shift-
ing this computation offline, we remove redundant
per-query processing at inference time.
3.3 Online Retrieval-Augmented In-Context
Learning
At inference, each query transaction xqis evaluated
through a retrieval-augmented reasoning pipeline
that casts fraud detection as case-based in-context
learning for LLMs. It consists of two stages: in-
stance retrieval and prompt-based reasoning.
1. Instance Retrieval:We partition the reduced
feature set Fselinto categorical Fcatand numeric
Fnumsubsets, where categorical attributes anchor
retrieval to structurally similar cases, and numeric
attributes enable fine-grained similarity search.
Categorical filtering. LetFcat={f (1), . . . , f (C)}
denote the selected categorical attributes sorted by
importance (high to low). We build the candidate
pool by progressively adding equality constraints
and backing off if the intersection becomes empty:
C(j)={x∈ C(j−1)|x[f (j)] =x q[f(j)]}, j= 1, . . . , C.
(2)
where C(0)=D ext. We stop at the largest j⋆such
thatC(j⋆)̸=∅as the final candidate setC(x q).Numeric similarity search. Within the candidate
pool, we retrieve neighbors based on cosine simi-
larity in the feature space (constructed offline):
s(x, x q) =⟨z(x), z(x q)⟩
∥z(x)∥ 2· ∥z(x q)∥2, x∈ C(x q).(3)
The top- nsimilar transactions constitute the re-
trieval set Dretrieved (xq). The hybrid retrieval strat-
egy combines semantic consistency (via categorical
filtering) with graded similarity (via vector search),
aligning the selection with the LLM’s strength at
analogical reasoning (Yasunaga et al., 2024).
2. Prompt Construction and In-Context Learn-
ing:Each retrieved example x∈ D retrieved (xq)
is converted into a descriptive, human-readable
sentence that covers only the features in Fsel, fol-
lowed by its ground-truth label expressed in natu-
ral language (“It is a fraud.” / “It is not a fraud.”).
The query case is serialized in the same schema
but presented without a label. Rather than list-
ing key–value pairs, we embed all features into a
natural-language template. The final prompt pro-
vided to the LLM has three components (using the
CCFRAUDdataset prompt as an example):
•Task instruction: “You are given several similar histor-
ical cases with their ground truth labels. Use them as
guidance and then assess the current case.”
•Few-shot examples: “Example 1: The client is a {},
the state number is {}, the number of cards is {}, . . . It
is a fraud...”
•Query: “Current case: The client is {}, the state number
is {}, the number of cards is...”
The prompt comprises the task instruction, a
series of few-shot examples with the query transac-
tion. By contextualizing structured feature values
in natural language, we leverage LLMs to inter-
pret tabular data and to draw analogies between the
query and retrieved transactions. This formulation
reframes fraud detection as case-based reasoning
rather than directly classifying tabular inputs.
4 Experiments
We answer the following research questions (RQ):
•RQ1: How do modern LLMs perform on tabular
financial fraud detection relative to traditional
machine-learning baselines, and does FinFRE-
RAG improve their performance?
•RQ2: How many features or relevant transac-
tions are needed for LLMs to understand and
detect the fraud?
•RQ3: What is the contribution of FinFRE-RAG’s
feature reduction module to overall performance?
4

•RQ4: How does the granularity of the LLM’s
output affect fraud detection performance?
•RQ5:Can FinFRE-RAG outperform LLMs fine-
tuned on fraud datasets?
4.1 Experiment Setup
Datasets.We evaluate on four public fraud detec-
tion datasets:CCF(Feng et al., 2023),CCFRAUD
(Feng et al., 2023; Kamaruddin and Ravi, 2016),
IEEE-CIS (Howard and Bouchon-Meunier, 2019),
and PAYSIM(Lopez-Rojas, 2017), covering real-
world and synthetic transaction data with varying
sizes, feature spaces, and fraud ratios. Detailed
dataset descriptions are provided in the Appendix
A. To ensure computational feasibility for inference
while preserving class imbalance, we randomly
sample 8,000 transactions for testing and 2,000 for
validation, maintaining the original fraud ratio. The
remaining transactions constitute the retrieval pool
(external datasetD ext) during inference.
Models.Considering the sensitive nature of fi-
nancial data, which often must remain strictly in-
house for compliance, we focus on open-weight
instruction-tuned LLMs that can be deployed on-
premise. Concretely, we evaluateQwen3-14B
andQwen3-Next-80B-A3B-Thinking(Yang et al.,
2025a),Gemma 3-12BandGemma 3-27B(Team
et al., 2025), as well asGPT-OSS-20BandGPT-
OSS-120B(Agarwal et al., 2025). These models
represent a range of medium to large instruction-
tuned LLMs. As non-LLM baselines for tabular
fraud detection, we includeRandom Forest,XG-
Boost, and the recent deep tabular modelTabM
(Gorishniy et al., 2024), covering widely used tree
ensembles and a strong state-of-the-art neural base-
line. There are also some domain -specific financial
LLMs, such as FinGPT (Yang et al., 2023) and
FinMA (Xie et al., 2023). However, these models
are based on earlier Llama versions (Touvron et al.,
2023b,a) and are comparatively smaller, which may
weaker instruction-following and reasoning for our
setting, so we do not include them in experiments.
Evaluation Metrics.Due to the class imbalance,
accuracy is not informative. We therefore use F1-
score and Matthews Correlation Coefficient (MCC)
(Chicco and Jurman, 2020) as primary metrics, as
they better reflect performance under imbalance.
We also report precision (share of flagged transac-
tions that are truly fraudulent) and recall (share of
frauds correctly identified) for analyst reference.
Implementation Details.Experiments are run on
4×A100 GPUs. Results reported are an average ofthree runs. Unless otherwise specified, we retain
the top k= 10 features and retrieve n= 20 near-
est neighbors to construct the in-context examples
(Section 4.2 analyzes sensitivity to kandn). For
supervised baselines, we train on each dataset’s
external splitusing all features and perform hy-
perparameter optimization (detailed in Appendix
Table 6) on the validation set for 50 trials. Instead
of letting LLMs directly output binary predictions,
we consider a 5-point risk score, where Score≥4
is viewed as fraud (positive). We present the full
prompts we used in Appendix C.
We finetuneQwen3 -14B,Gemma 3 -12B, and
GPT -OSS-20Bvia LoRA (Hu et al., 2022) using
UNSLOTH(Daniel Han and team, 2023), on the
external splitonly. Since they are tuned on the ex-
ternal split, we do not apply RAG on these models;
instead, we directly run inference for each transac-
tion. We conduct a small grid over key hyperparam-
eters on 10% of the data to select rank, α, learning
rate, and learning rate schedulers, then train for one
epoch. Full finetuning details (adapter targets, rank,
learning rates, and batch size) are in Appendix B.
4.2 Results
RQ1: Baseline LLM performance and impact
of FinFRE -RAG.We present the main compari-
son results in Table 1, where we applied the same
prompt templates as FinFRE-RAG. The key differ-
ence is that baseline LLMs only use the query trans-
action without retrieval-augmented generation.
Baseline performance.The results for base-
line LLMs are consistent with findings in (Xie
et al., 2024). LLMs perform poorly when directly
prompted to classify fraud data without additional
context, and they are substantially below strong tab-
ular methods likeTabM. For instance,Qwen3 -14B
achieves a negative MCC score on theCCF, indicat-
ing that the model’s predictions on this dataset are
indistinguishable from random guessing. Similar
trends hold for other models. These results confirm
that LLMs, when applied to challenging financial
tabular data, fail to capture fraud patterns.
FinFRE -RAG performance.Incorporating
FinFRE -RAG with existing LLMs dramatically
improves their performance across all datasets.
Qwen3 -14B’s F1 -scores increase from 0.00 to 0.31
onCCF, from 0.14 to 0.48 onCCFRAUD, from 0.04
to 0.62 on IEEE -CIS, and from 0.00 to 0.11 on
PAYSIM. Correspondingly, MCC rises from neg-
ative or near -zero values to 0.36, 0.46, 0.60, and
0.22, respectively. All of the other models benefit
5

ModelCCF CCFRAUDIEEE-CIS PAYSIM
F1 MCC Prec. Rec. F1 MCC Prec. Rec. F1 MCC Prec. Rec. F1 MCC Prec. Rec.
Qwen3-14B 0.00 -0.01 0.00 0.00 0.14 0.09 0.07 0.80 0.04 -0.01 0.03 0.05 0.00 -0.05 0.00 0.46
+FinFRE-RAG 0.31 0.36 0.20 0.64 0.48 0.46 0.59 0.41 0.62 0.60 0.61 0.63 0.11 0.22 0.06 0.85
Qwen3-Next-80B 0.01 0.07 0.01 0.85 0.12 0.04 0.06 0.99 0.08 0.05 0.04 0.77 0.00 0.00 0.00 1.00
+FinFRE-RAG 0.08 0.19 0.04 0.86 0.48 0.45 0.43 0.55 0.35 0.38 0.23 0.75 0.09 0.19 0.05 0.85
Gemma 3-12B 0.00 0.00 0.00 0.07 0.13 0.09 0.07 0.97 0.01 -0.03 0.01 0.01 0.00 0.01 0.00 1.00
+FinFRE-RAG 0.79 0.80 0.68 0.93 0.59 0.57 0.60 0.59 0.59 0.57 0.59 0.59 0.71 0.72 0.61 0.85
Gemma 3-27B 0.00 0.01 0.00 0.43 0.12 0.05 0.06 0.99 0.03 -0.02 0.02 0.05 0.00 0.01 0.00 1.00
+FinFRE-RAG 0.79 0.78 0.79 0.79 0.55 0.53 0.57 0.54 0.64 0.63 0.66 0.63 0.73 0.74 0.65 0.85
GPT-OSS-20B 0.02 0.04 0.01 0.21 0.12 0.04 0.07 0.83 0.03 0.00 0.03 0.03 0.00 -0.03 0.00 0.61
+FinFRE-RAG 0.24 0.32 0.15 0.71 0.51 0.49 0.54 0.49 0.61 0.59 0.54 0.68 0.17 0.28 0.10 0.85
GPT-OSS-120B 0.01 0.06 0.01 0.64 0.14 0.08 0.08 0.77 0.04 -0.01 0.03 0.08 0.00 0.02 0.00 1.00
+FinFRE-RAG 0.44 0.46 0.33 0.64 0.53 0.51 0.63 0.46 0.66 0.64 0.64 0.68 0.25 0.36 0.15 0.85
Random Forest 0.85 0.85 0.92 0.78 0.52 0.52 0.39 0.82 0.55 0.54 0.46 0.68 0.79 0.81 0.65 1.00
XGBoost 0.89 0.89 0.92 0.86 0.48 0.50 0.32 0.92 0.74 0.73 0.67 0.82 0.68 0.71 0.55 0.92
TabM 0.85 0.85 0.92 0.79 0.66 0.65 0.71 0.62 0.82 0.82 0.88 0.76 0.92 0.92 1.00 0.85
Table 1: Performance comparison of baseline models and FinFRE -RAG on fraud datasets. Each block reports F1,
MCC, Precision, and Recall; higher scores indicate better performance.
from the FinFRE -RAG and significantly achieve
better MCC. Some results are comparable or even
exceed the training-based methods likeRandom
ForestandXGBoost. These gains illustrate that
FinFRE -RAG effectively mitigates the difficulties
LLMs face when reasoning over raw tabular in-
puts. By restricting the prompt to a small set of
informative features and providing relevant histori-
cal examples, the LLM learns to map numeric and
categorical patterns to fraud risk scores.
5 10 15 20 25
K0.20.30.40.50.60.70.8MCC
Qwen3-14B
Gemma 3-12B
GPT-OSS-20B
(a)CCF
5 10 15 20 25
K0.400.450.500.550.60MCC
Qwen3-14B
Gemma 3-12B
GPT-OSS-20B (b) IEEE-CIS
Figure 2: MCC vs. number of selected featuresk.
RQ2: How many features or relevant transac-
tions are needed for LLMs to understand and
detect the fraud?We first conduct experiments
to study the impacts of the feature-reduction pa-
rameter kand the number of retrieved transactions
n. We vary the number of retained attributes from
5 to 25 and report MCC on theCCFand IEEE-
CIS datasets, both of which contain more than 25
features. As shown in Figure 2, simply retaining
more features does not necessarily translate into im-
proved performance. Instead,the models achieve
their best results when provided with a compact
yet informative subset of high-impact attributes.
5 10 15 20 25
N0.20.40.60.8MCC
Qwen3-14B
Gemma 3-12B
GPT-OSS-20B(a)CCF
5 10 15 20 25
N0.520.540.560.580.600.62MCC
Qwen3-14B
Gemma 3-12B
GPT-OSS-20B (b) IEEE-CIS
Figure 3: MCC vs. number of retrieved transactionsn.
This finding underscores the importance of control-
ling feature dimensionality to avoid overwhelming
the LLM with irrelevant inputs.
We next examine the effect of the number of
retrieved transactions nused for in-context ex-
amples (Figure 3). Two general trends emerge.
First, reasoning models, such asQwen3 -14Band
GPT -OSS-20B, consistently benefit from larger re-
trieval sets. On IEEE-CIS, for example,Qwen3-
14Bimproves from an MCC of 0.3 to 0.45 as n
increases, whileGPT-OSS-20Brises from 0.34 to
0.43. Similar gains are observed onCCF. These
improvements suggest that larger retrieval sets en-
able reasoning models to extract more nuanced
relationships between features with more historical
cases. In contrast,Gemma 3, as a non-reasoning
model, is more sensitive to nand exhibits dimin-
ishing gains when too many examples are included.
To further investigatewhy additional retrieved
samples may degradeGemma-3’s performance,
we analyzed its prediction behavior and found that
Gemma-3often relies on the aggregate distribution
of retrieved examples, particularly the proportion
of fraudulent (positive) transactions, as a primary
6

Original Random
Dataset F1 MCC F1 MCC
CCF 0.31 / 0.79 / 0.24 0.36 / 0.80 / 0.32 0.26 / 0.64 / 0.21 0.30 / 0.65 / 0.28
IEEE-CIS 0.62 / 0.59 / 0.61 0.60 / 0.57 / 0.59 0.27 / 0.31 / 0.34 0.32 / 0.37 / 0.32
Table 2: Feature selection ablation study. Each cell lists
the values in order Qwen/Gemma/GPT-OSS.
signal for classification. In comparison,Qwen3-
14BandGPT-OSS-20Bconsider both the number
of positive instances and feature-level differences
between the query and the retrieved samples, rea-
soning about how these variations relate to fraud
likelihood. Consequently, when the retrieval pool
forGemma-3includes more heterogeneous or con-
tradictory examples (e.g., less-similar transactions
with opposite labels), its decision confidence de-
clines, leading to poorer performance.
RQ3: Contribution of importance-guided
feature reduction.We compare FinFRE-RAG
against a variant that retains the same number of
attributes chosen uniformly at random. Because
CCFRAUDand PAYSIMhave too few attributes
for meaningful ranking, this ablation is still con-
ducted onCCFand IEEE-CIS only. Table 2 sum-
marizes results for the three models (Qwen3 -14B,
Gemma3 -12B, andGPT -OSS-20B). In all cases, se-
lecting features by importance consistently outper-
forms random selection in both F1 and MCC. The
effect is especially pronounced on IEEE-CIS. This
dataset includes hundreds of numerical and categor-
ical features, making it more likely that random se-
lection captures irrelevant or redundant signals that
dilute the predictive context provided to the LLM.
By contrast,CCFhas only 30 attributes, reducing
the risk of severe noise introduction when sampling
at random. Consequently, performance degradation
is larger on IEEE-CIS than onCCF. These results
highlight the value of prioritizing the most informa-
tive attributes. By focusing the model’s reasoning
on high-signal features, importance-guided reduc-
tion enhances the reliability of retrieval-augmented
fraud detection, ensuring that the LLM is not over-
whelmed by noisy or extraneous inputs.
RQ4: Comparing the granularity of LLM’s out-
put.Prior studies typically frame LLM -based
fraud detection as a direct binary classification task:
the model is asked to decide whether a transaction
is “bad” or “good”. In this work, we instead ask
the LLM to produce a 5 -point risk score: Score 1
indicating the lowest probability of fraud and Score
5 the highest, along with a brief explanation, for aScoring Binary
Dataset F1 MCC F1 MCC
CCF 0.31 /0.79 / 0.240.36 /0.80 / 0.32 0.49/ 0.26 / 0.100.51/ 0.37 / 0.18
CCFRAUD 0.48 /0.59 / 0.510.46 /0.57 / 0.49 0.5/ 0.53 / 0.400.47/ 0.51 / 0.37
IEEE-CIS 0.62 / 0.59 / 0.61 0.60 / 0.57 / 0.59 0.51 / 0.33 / 0.42 0.51 / 0.36 / 0.40
PAYSIM 0.11 / 0.71 / 0.17 0.22 / 0.72 / 0.28 0.07 / 0.26 / 0.15 0.17 / 0.34 / 0.26
Table 3: Risk scoring vs. binary output. Each cell lists
the values in order Qwen/Gemma/GPT-OSS.
better interpretable generation to human analysts.
However, it is not obvious whether increasing out-
put granularity enhances or hinders detection accu-
racy. To answer this question, we compare these
two prompting strategies under the same setting.
Table 3 presents F1 and MCC across four datasets.
Our results demonstrate that fine-grained scor-
ing yields benefits across most datasets and models.
Converting the decision into a 5-level risk score
improves both F1 and MCC relative to a binary
response in most cases. These results suggest that
risk scoring allows models to leverage uncertainty
more effectively, extracting additional signal from
ambiguous cases that would otherwise be forced
into a hard binary decision. Overall, we find that
risk scoring not only yields more informative out-
put but also improves detection performance.
RQ5: Can fine-tuning LLMs yield better de-
tections?We adopted two prompting regimes:
descriptiveandschema-grounded(Appendix C) to
fine-tune three LLMs. Concretely, we use two dif-
ferent types of datasets, the simulated (CCFRAUD)
and the real-world (IEEE-CIS), withdescriptive
andschema-groundedprompts, respectively. Mod-
els are fine-tuned to predict fraud labels from the
formatted inputs. Since these datasets contain only
binary labels, to keep training–evaluation consis-
tent, we restrict outputs to binary classification
without explanations.
ModelCCFRAUDIEEE-CIS
F1 MCC Prec. Rec. F1 MCC Prec. Rec.
Qwen3-14B w/ ours0.48 0.46 0.59 0.41 0.62 0.60 0.61 0.63
Qwen3-14B FT0.55 0.52 0.62 0.49 0.52 0.53 0.71 0.41
Gemma 3-12B w/ ours0.59 0.57 0.60 0.59 0.59 0.57 0.59 0.59
Gemma 3-12B FT0.57 0.55 0.48 0.69 0.40 0.44 0.72 0.28
GPT-OSS-20B w/ ours0.51 0.49 0.54 0.49 0.61 0.59 0.54 0.68
GPT-OSS-20B FT0.61 0.58 0.62 0.60 0.45 0.47 0.68 0.34
Table 4: Finetuned LLMs (FT) vs. FinFRE-RAG (ours).
Direct fine-tuning on task data doesnotcon-
sistently outperform FinFRE-RAG. Across our
settings, FT models tend to exhibit higher preci-
sion but lower recall than FinFRE-RAG, indicating
potential under-coverage of diverse fraud modes
and/or overfitting to the majority class in imbal-
anced data. By contrast, FinFRE-RAG maintains
7

GPT-OSS-20B GPT-OSS-20B with FinFRE -RAGYou are a helpful financial expert …...Please give a range of 1 to 5 for the probability of fraud …….The transaction has attributes: TransactionDT: 10372086, 
TransactionAmt: 12.205, card1: 2801, C13: 0.000, card2: 130.000, C14: 0.000, addr1: missing, P_emaildomain: gmail.com, card6:  credit, C1: 3.000.Transaction Description of a Fraud Case
Score: 2  – The transaction amount (≈ $12) is low, which is typical for many 
legitimate purchases.  The email domain is a common, reputable provider 
(gmail.com).  The card type is a standard credit card (card6 = credit).  The only 
notable red flag is the missing addr1 value, which can sometimes indicate 
incomplete data but on its own is not a strong fraud indicator.Overall, the 
limited risk factors suggest a relatively low probability of fraud, though the 
missing address keeps it from being the absolute lowest score.Score: 4 – The transaction shares many key fraud -signatures seen in the 
training set: it uses a Gmail address, a credit card, a small amount (~$12), and a 
high C1 value (3.0). Most Gmail -credit transactions with C1 ≥ 2 in the examples 
were labeled fraud (e.g., examples 1, 4, 9, 11, 16). The only notable differences 
are the relatively low card1 value (2801) and the missing address field, but 
these alone have not been strong enough to override the other strong fraud 
indicators. Thus the probability of fraud is high, though not absolute.Figure 4: Example responses from GPT-OSS-20B without and with FinFRE-RAG. With in-context learning on
similar transactions, it learns to use patterns between different transactions to identify potential fraud.
a more balanced precision–recall profile, likely be-
cause retrieval exposes the model to heterogeneous,
label-aware exemplars at inference. These observa-
tions underscore that instance-grounded reasoning,
rather than parameter updates alone, is a key driver
of robustness under class imbalance. Moreover,
because tabular datasets contain limited natural
language, pure fine-tuning biases models toward
narrow binary prediction heads, diminishing their
general generative capacity and precluding case-
specific explanations. At the current stage, ICL
with retrieval offers a more practical pathway than
fine-tuning for financial fraud detection. Advanc-
ing domain-specific fine-tuning methods that pre-
serve reasoning and explanation on structural data
remains an important direction for future work.
4.3 Further analysis and discussion
Behavioral differences.To illustrate the be-
havioral differences induced by our framework,
we presentGPT-OSS-20B’s responses with and
without FinFRE-RAG on a fraudulent transaction.
When directly making the prediction (Figure 4
left), the model relies predominantly on broad, pre-
trained priors and surface correlations (e.g., treat-
ing the email domain as a direct cue), yielding
judgments that are weakly grounded in the tabu-
lar evidence. In contrast, with FinFRE-RAG the
model conditions on retrieved, label-aware exem-
plars and an importance-guided subset of attributes,
and its rationale explicitly reflects cross-feature in-
teractions. Concretely, FinFRE-RAG response cal-
ibrates risk by aligning the query against historical
cases and by weighing feature-level deltas, produc-
ing explanations that mirror dataset-specific fraud
regularities rather than spurious textual heuristics.
Why Gemma 3 outperforms others significantly
onCCFand PAYSIM?As we stated in RQ2 it
relies on the label distribution within the retrieved
neighbors. On PAYSIM(synthetic) andCCF(withPCA-compressed features), cosine similarity over
the feature set yields high-purity neighborhoods:
retrieved examples are close and mostly share the
same label. Gemma 3 finds a shortcut, leaning on
a simple majority signal in the neighborhood to
produce high-recall predictions. By contrast, rea-
soning models sometimes “over-interpret” benign
within-cluster differences, which can slightly lower
F1 under such clustered conditions. Therefore, in
the real-world IEEE-CIS, where neighborhoods are
more mixed (heterogeneous labels, cross-feature
confounds), we findGemma 3cannot outperform
other models without finer cross-reasoning.
PAYSIMrecall uniformity(bias from label homo-
geneous neighborhoods). The test split of PAYSIM
contains only 13 positive cases; for 11 of them the
retrieved neighbors were all positive, and for the
remaining 2 they were all negative. Because the
LLMs often rely on the majority label within the
retrieved set, these label -homogeneous neighbor-
hoods produce near -deterministic outcomes and
explain the uniform recall in Table 1. This exposes
a weakness of our method: when neighborhoods
are label -biased, predictions can be skewed by re-
trieval rather than feature-level evidence.
5 Conclusion
We introduced FinFRE -RAG, a two -stage frame-
work, converting selected numerical and categor-
ical attributes into descriptive natural language
and grounding each prediction in similar historical
cases, to adapt LLMs for fraud detection in tabular
financial transactions. FinFRE -RAG substantially
improves LLM performance over direct prompting,
making it a promising direction for future research
and production. Based on the framework, we fur-
ther examine key factors impacting LLMs’ under-
standing of fraud. Overall, FinFRE-RAG provides
a practical path toward reliable, transparent fraud
assessments and opens avenues for future research.
8

Limitations
Our study has several limitations that suggest cau-
tion and opportunities for future work.
•Model and baseline scope.We evaluate a
representative set of open-weight LLMs and
common tabular baselines, but do not compare
against production-grade systems with large fea-
ture stores, domain-specialized LLMs, or very
large models.
•Datasets and validity.For compliance reasons,
we rely exclusively on public datasets. Only
IEEE-CIS contains a rich set of features; others
have relatively few and function more like “toy”
datasets (e.g.,CCFRAUD, PAYSIM). This under-
represents real-world settings where volume and
feature cardinality are much higher, patterns are
subtler, and fraud is harder to capture. Moreover,
IEEE-CIS andCCFanonymize feature names;
in practice, revealing semantically meaningful
feature names to the LLM could further improve
reasoning, which we are not able to evaluate here.
•Metrics and systems considerations.Our eval-
uation emphasizes standard classification met-
rics (F1/MCC). Because the study is research-
oriented and our system is not production-tuned,
we did not optimize for or report latency, through-
put, or serving cost under realistic traffic.
•Explanations vs. causality.While we analyze
model rationales qualitatively, these are heuris-
tic and conditioned on retrieved exemplars; they
should not be interpreted as causal explanations.
•Retrieval design.We use cosine similarity
over importance-guided features, for example
retrieval. Alternative distance metrics, learned re-
trievers, or more advanced RAG pipelines could
yield stronger neighborhoods and further gains.
•Human evaluation.Beyond a few illustrative
case studies, we lack systematic human reviews.
Pattern attributions produced by our pipeline re-
quire additional validation by human reviewers.
•Domain breadth.This work is a case study
infinancial fraud. Whether our method can be
transferred to other domains remains to be tested.
•Fine-tuning budget.We employ LoRA-style
parameter-efficient fine-tuning and do not fine-
tune larger models. This choice may cap achiev-
able performance and limit conclusions.
Ethical Considerations
•Data privacy and security.Fraud datasets may
contain PII and sensitive attributes. We recom-mend strictly on-premise deployment with open-
weight models whenever feasible, minimizing
data exfiltration risk. Retrieval stores should be
privacy-hardened to prevent sensitive attributes
from leaking through prompts or rationales.
•Transparency and accountability.FinFRE-
RAG produces concise rationales, but these are
model-generated and may omit relevant factors.
Institutions should treat them asdecision sup-
port, not final determinations. Deployed systems
should maintain model cards, data provenance
records, retrieval policies, and reviewer notes,
and provide users with appropriate notices/ex-
planations for adverse actions. Human review
should be mandatory for high-impact outcomes.
•Misuse and overreliance.Because LLM out-
puts can appear authoritative, there is a risk of
over-trust. We caution against fully autonomous
blocking and recommend conservative thresh-
olds, analyst-in-the-loop workflows, and defense-
in-depth with orthogonal detectors. Systems
should be stress-tested for prompt injection and
retrieval poisoning, protected by robust access
and usage controls, and monitored for distribu-
tional shifts that could degrade performance or
fairness over time.
References
Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Alt-
man, Andy Applebaum, Edwin Arbus, Rahul K
Arora, Yu Bai, Bowen Baker, Haiming Bao, and 1
others. 2025. gpt-oss-120b & gpt-oss-20b model
card.arXiv preprint arXiv:2508.10925.
Toyin D Aguda, Suchetha Siddagangappa, Elena Kochk-
ina, Simerjot Kaur, Dongsheng Wang, and Charese
Smiley. 2024. Large language models as finan-
cial data annotators: A study on effectiveness and
efficiency. InProceedings of the 2024 Joint In-
ternational Conference on Computational Linguis-
tics, Language Resources and Evaluation (LREC-
COLING 2024), pages 10124–10145.
Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru
Ohta, and Masanori Koyama. 2019. Optuna: A next-
generation hyperparameter optimization framework.
InProceedings of the 25th ACM SIGKDD interna-
tional conference on knowledge discovery & data
mining, pages 2623–2631.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, and 1 others. 2020. Language models are
few-shot learners.Advances in neural information
processing systems, 33:1877–1901.
9

Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A
scalable tree boosting system. InProceedings of
the 22nd acm sigkdd international conference on
knowledge discovery and data mining, pages 785–
794.
Davide Chicco and Giuseppe Jurman. 2020. The advan-
tages of the matthews correlation coefficient (mcc)
over f1 score and accuracy in binary classification
evaluation.BMC genomics, 21(1):6.
Michael Han Daniel Han and Unsloth team. 2023. Un-
sloth.
Yingtong Dou, Zhiwei Liu, Li Sun, Yutong Deng, Hao
Peng, and Philip S Yu. 2020. Enhancing graph neural
network-based fraud detectors against camouflaged
fraudsters. InProceedings of the 29th ACM inter-
national conference on information & knowledge
management, pages 315–324.
Duanyu Feng, Yongfu Dai, Jimin Huang, Yifang Zhang,
Qianqian Xie, Weiguang Han, Zhengyu Chen, Ale-
jandro Lopez-Lira, and Hao Wang. 2023. Empow-
ering many, biasing a few: Generalist credit scor-
ing through large language models.arXiv preprint
arXiv:2310.00566.
Ugo Fiore, Alfredo De Santis, Francesca Perla, Paolo
Zanetti, and Francesco Palmieri. 2019. Using genera-
tive adversarial networks for improving classification
effectiveness in credit card fraud detection.Informa-
tion Sciences, 479:448–455.
João Gama, Indr ˙e Žliobait ˙e, Albert Bifet, Mykola Pech-
enizkiy, and Abdelhamid Bouchachia. 2014. A sur-
vey on concept drift adaptation.ACM computing
surveys (CSUR), 46(4):1–37.
Yury Gorishniy, Akim Kotelnikov, and Artem Babenko.
2024. Tabm: Advancing tabular deep learning
with parameter-efficient ensembling.arXiv preprint
arXiv:2410.24210.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
Waleed Hilal, S Andrew Gadsden, and John Yawney.
2022. Financial fraud: a review of anomaly detection
techniques and recent advances.Expert systems With
applications, 193:116429.
Addison Howard and Bernadette Bouchon-Meunier.
2019. Ieee cis, inversion, john lei, lynn@ vesta, mar-
cus2010, hussein abbass.IEEE-CIS Fraud Detection,
Kaggle.
Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu
Chen. 2022. LoRA: Low-rank adaptation of large
language models. InInternational Conference on
Learning Representations.Jun Hu, Wenwen Xia, Xiaolu Zhang, Chilin Fu, We-
ichang Wu, Zhaoxin Huan, Ang Li, Zuoli Tang, and
Jun Zhou. 2024. Enhancing sequential recommenda-
tion via llm-based semantic embedding learning. In
Companion Proceedings of the ACM Web Conference
2024, pages 103–111.
Tairan Huang and Yili Wang. 2025. Can llms find fraud-
sters? multi-level llm enhanced graph fraud detection.
arXiv preprint arXiv:2507.11997.
Jing Jin and Yongqing Zhang. 2025. The analysis of
fraud detection in financial market under machine
learning.Scientific Reports, 15(1):29959.
SK Kamaruddin and Vadlamani Ravi. 2016. Credit
card fraud detection using big data analytics: use of
psoaann based one-class classification. InProceed-
ings of the international conference on informatics
and analytics, pages 1–8.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1), pages 6769–6781.
Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang,
Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu.
2017. Lightgbm: A highly efficient gradient boost-
ing decision tree.Advances in neural information
processing systems, 30.
Seunghee Kim, Changhyeon Kim, and Taeuk Kim. 2025.
Fcmr: Robust evaluation of financial cross-modal
multi-hop reasoning. InProceedings of the 63rd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 23352–
23380, Vienna, Austria. Association for Computa-
tional Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Haohang Li, Yupeng Cao, Yangyang Yu, Shashid-
har Reddy Javaji, Zhiyang Deng, Yueru He, Yuechen
Jiang, Zining Zhu, K.p. Subbalakshmi, Jimin Huang,
Lingfei Qian, Xueqing Peng, Jordan W. Suchow, and
Qianqian Xie. 2025. Investorbench: A benchmark
for financial decision-making tasks with llm-based
agent. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 2509–2525, Vienna,
Austria. Association for Computational Linguistics.
Kaidi Li, Tianmeng Yang, Min Zhou, Jiahao Meng,
Shendi Wang, Yihui Wu, Boshuai Tan, Hu Song, Lu-
jia Pan, Fan Yu, and 1 others. 2024. Sefraud: Graph-
based self-explainable fraud detection via interpreta-
tive mask learning. InProceedings of the 30th ACM
SIGKDD Conference on Knowledge Discovery and
Data Mining, pages 5329–5338.
10

E Lopez-Rojas. 2017. Synthetic financial datasets
for fraud detection.Kaggle. Available online:
https://www. kaggle. com/datasets/ealaxi/paysim1
(accessed on 29 July 2023).
Junyu Luo, Zhizhuo Kou, Liming Yang, Xiao Luo,
Jinsheng Huang, Zhiping Xiao, Jingshu Peng,
Chengzhong Liu, Jiaming Ji, Xuanzhe Liu, Sirui Han,
Ming Zhang, and Yike Guo. 2025. Finmme: Bench-
mark dataset for financial multi-modal reasoning eval-
uation. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 29465–29489, Vienna,
Austria. Association for Computational Linguistics.
Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe,
Mike Lewis, Hannaneh Hajishirzi, and Luke Zettle-
moyer. 2022. Rethinking the role of demonstra-
tions: What makes in-context learning work?arXiv
preprint arXiv:2202.12837.
Yansong Ning, Shuowei Cai, Wei Li, Jun Fang,
Naiqiang Tan, Hua Chai, and Hao Liu. 2025. Dima:
An llm-powered ride-hailing assistant at didi. InPro-
ceedings of the 31st ACM SIGKDD Conference on
Knowledge Discovery and Data Mining V . 2, pages
4728–4739.
Liudmila Prokhorenkova, Gleb Gusev, Aleksandr
V orobev, Anna Veronika Dorogush, and Andrey
Gulin. 2018. Catboost: unbiased boosting with cat-
egorical features.Advances in neural information
processing systems, 31.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models.Transactions of the Association for
Computational Linguistics, 11:1316–1331.
Gurjot Singh, Prabhjot Singh, and Maninder Singh.
2025. Advanced real-time fraud detection using rag-
based llms.arXiv preprint arXiv:2501.15290.
Makram Soui, Ines Gasmi, Salima Smiti, and Khaled
Ghédira. 2019. Rule-based credit risk assessment
model using multi-objective evolutionary algorithms.
Expert systems with applications, 126:144–157.
Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya
Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin,
Tatiana Matejovicova, Alexandre Ramé, Morgane
Rivière, and 1 others. 2025. Gemma 3 technical
report.arXiv preprint arXiv:2503.19786.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, and 1 others. 2023a. Llama: Open and ef-
ficient foundation language models.arXiv preprint
arXiv:2302.13971.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, ShrutiBhosale, and 1 others. 2023b. Llama 2: Open foun-
dation and fine-tuned chat models.arXiv preprint
arXiv:2307.09288.
Chengrui Wang, Qingqing Long, Meng Xiao, Xunxin
Cai, Chengjun Wu, Zhen Meng, Xuezhi Wang, and
Yuanchun Zhou. 2024. Biorag: A rag-llm framework
for biological question reasoning.arXiv preprint
arXiv:2408.01107.
Jianling Wang, Yifan Liu, Yinghao Sun, Xuejian Ma,
Yueqi Wang, He Ma, Zhengyang Su, Minmin Chen,
Mingyan Gao, Onkar Dalal, and 1 others. 2025. User
feedback alignment for LLM-powered exploration
in large-scale recommendation systems. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 6: Industry
Track), pages 996–1003.
Neng Wang, Hongyang Yang, and Christina Wang. Fin-
gpt: Instruction tuning benchmark for open-source
large language models in financial datasets. In
NeurIPS 2023 Workshop on Instruction Tuning and
Instruction Following.
Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski,
Mark Dredze, Sebastian Gehrmann, Prabhanjan Kam-
badur, David Rosenberg, and Gideon Mann. 2023.
Bloomberggpt: A large language model for finance.
arXiv preprint arXiv:2303.17564.
Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu
Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong
Li, Yongfu Dai, Duanyu Feng, and 1 others. 2024.
Finben: A holistic financial benchmark for large lan-
guage models.Advances in Neural Information Pro-
cessing Systems, 37:95716–95743.
Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao
Lai, Min Peng, Alejandro Lopez-Lira, and Jimin
Huang. 2023. Pixiu: A comprehensive benchmark,
instruction dataset and large language model for fi-
nance.Advances in Neural Information Processing
Systems, 36:33469–33484.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025a. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Chengdong Yang, Hongrui Liu, Daixin Wang, Zhiqiang
Zhang, Cheng Yang, and Chuan Shi. 2025b. Flag:
Fraud detection with llm-enhanced graph neural net-
work. InProceedings of the 31st ACM SIGKDD
Conference on Knowledge Discovery and Data Min-
ing V . 2, pages 5150–5160.
Hongyang Yang, Xiao-Yang Liu, and Christina Dan
Wang. 2023. Fingpt: Open-source financial large lan-
guage models.FinLLM Symposium at IJCAI 2023.
Shu Yang, Shenzhe Zhu, Zeyu Wu, Keyu Wang, Junchi
Yao, Junchao Wu, Lijie Hu, Mengdi Li, Derek F
Wong, and Di Wang. 2025c. Fraud-r1: A multi-
round benchmark for assessing the robustness of llm
11

against augmented fraud and phishing inducements.
arXiv preprint arXiv:2502.12904.
Yi Yang, Mark Christopher Siy UY , and Allen
Huang. 2020. Finbert: A pretrained language
model for financial communications.Preprint,
arXiv:2006.08097.
Michihiro Yasunaga, Xinyun Chen, Yujia Li, Panupong
Pasupat, Jure Leskovec, Percy Liang, Ed H. Chi,
and Denny Zhou. 2024. Large language models as
analogical reasoners. InThe Twelfth International
Conference on Learning Representations.
Jianke Yu, Hanchen Wang, Xiaoyang Wang, Zhao Li,
Lu Qin, Wenjie Zhang, Jian Liao, and Ying Zhang.
2023. Group-based fraud detection network on e-
commerce platforms. InProceedings of the 29th
ACM SIGKDD conference on knowledge discovery
and data mining, pages 5463–5475.
Yangyang Yu, Zhiyuan Yao, Haohang Li, Zhiyang Deng,
Yuechen Jiang, Yupeng Cao, Zhi Chen, Jordan Su-
chow, Zhenyu Cui, Rong Liu, and 1 others. 2024.
Fincon: A synthesized llm multi-agent system with
conceptual verbal reinforcement for enhanced finan-
cial decision making.Advances in Neural Informa-
tion Processing Systems, 37:137010–137045.
12

Dataset# Numerical
Features# Categorical
Features# Positive # NegativeImbalanced
(%)License
CCF30 - 490 284,317 0.172 DbCL v1.0
CCFRAUD5 2 596,014 9,403,986 5.98 DbCL v1.0
IEEE-CIS373 20 20,663 569,877 3.50 Competition Data
PAYSIM5 3 8,213 6,354,407 0.13 CC BY-SA 4.0
Table 5: Dataset summary. “# ” denotes feature counts; dashes indicate not applicable.
A Datasets
We use four publicly available fraud detection
datasets (stats shown in Table 5) in our experiments.
Below, we provide detailed descriptions:
•CCF1(Feng et al., 2023): A credit card fraud
dataset with 284,807 transactions from European
cardholders in September 2013. Each record
includes 30 features, 28 of which are anonymized
via PCA for confidentiality. The dataset is highly
imbalanced, with only 0.172% fraudulent cases.
•CCFRAUD2(Feng et al., 2023; Kamaruddin and
Ravi, 2016): It is asimulateddataset contains
about 1 million transactions with 7 features (e.g.,
gender, account balance, transaction count). The
proportion of fraudulent samples is 5.98%.
•IEEE-CIS3(Howard and Bouchon-Meunier,
2019): IEEE-CIS is areal-worlddataset.
We use the transaction-level data, which con-
sists of 590,540 transactions with hundreds of
anonymized numerical and categorical features.
Fraudulent transactions make up 3.5% of the data,
presenting challenges in both dimensionality and
class imbalance. The license can be found on
Kaggle4.
•PAYSIM5(Lopez-Rojas, 2017): A large-scale
simulatedmobile money transaction dataset
with 6.3 million samples. Each transaction
is labeled as fraud or non-fraud, with fea-
tures such as transaction type, amount, and
balance updates. The fraud ratio is 0.13%.
As noted in the dataset documentation, some
1https://www.kaggle.com/datasets/mlg-
ulb/creditcardfraud
2https://github.com/The-FinAI/CALM/blob/main/data/fraud
detection/ccFraud/
3https://www.kaggle.com/competitions/ieee-fraud-
detection/overview
4https://www.kaggle.com/competitions/ieee-fraud-
detection/rules
5https://www.kaggle.com/datasets/ealaxi/paysim1columns ( oldbalanceOrg ,newbalanceOrig ,
oldbalanceDest ,newbalanceDest ) can lead to
information leakage, as models may exploit them
as shortcuts for fraud detection. We neverthe-
less include these features, since removing them
would leave very limited information for learning.
Therefore, evaluation results on PaySim should
be interpreted with caution, and more emphasis
should be placed on the other three datasets.
To ensure computational feasibility for LLM in-
ference while preserving class imbalance, we ran-
domly sample 8,000 transactions from each dataset
for testing and 2,000 for validation, maintaining
the original fraud ratio. The remaining transac-
tions constitute the retrieval pool used by FinFRE-
RAG during inference. Fraudulent transactions are
treated as positive samples in all evaluations. For
missing values, we mark them as "missing" in LLM
inferences.
B Hyperparameters
LLM inference.For LLM inference, we set the
LLM temperature to 0.6 to balance determinism
and diversity in rationales and nucleus sampling
to 0.95. The max length is set to 16,384. We
retain the top k= 10 features for datasets and
retrieve n= 20 nearest neighbors to construct
the in-context examples. For those datasets with
fewer than 10features, we use all features during
inference.
Baseline Training.We tune all supervised base-
lines with Optuna (Akiba et al., 2019), using50
trialsper dataset–model pair. Each trial is trained
onDext; the configuration with the best validation
MCC is selected and evaluated on the test set. For
TabM training, we deployed early stopping with
patience of 5 epochs. For tree methods, we set
class weights via scale_pos_weight (XGBoost)
orclass_weight=balanced (RF) to mitigate the
impact of class imbalance. Table 6 lists the search
spaces.
13

Model Hyperparameter Search Space (Optuna)
Random Forestn_estimatorssuggest_int(50, 200)
max_depthsuggest_int(5, 20)
min_samples_splitsuggest_int(2, 10)
min_samples_leafsuggest_int(1, 5)
max_featuressuggest_categorical([“sqrt”,“log2”,None])
bootstrapsuggest_categorical([True,False])
class_weight fixed to“balanced”
XGBoostn_estimatorssuggest_int(50, 200)
max_depthsuggest_int(3, 10)
learning_ratesuggest_float(0.01, 0.5)
subsamplesuggest_float(0.6, 1.0)
colsample_bytreesuggest_float(0.6, 1.0)
reg_alphasuggest_float(0, 10)
reg_lambdasuggest_float(0, 10)
scale_pos_weight fixed tor=#negative
#positive(training split)
TabMn_blockssuggest_int(2, 4)
d_blocksuggest_categorical([128, 256, 512])
ksuggest_categorical([16, 32, 64])
dropoutsuggest_float(0.0, 0.3)
learning_ratesuggest_float(1e-4, 1e-2, log=True)
embedding_method fixed to“piecewise linear embedding”
n_binssuggest_int(16, 48)
d_embeddingsuggest_categorical([4, 8, 16, 32])
Table 6: Optuna search spaces used in our experiments for all baselines.
LLM Fine-tuning.We apply parameter-efficient
finetuning (PEFT) with LoRA (Hu et al., 2022)
toQwen3-14B,Gemma 3-12B, andGPT-OSS-20B
using Unsloth (Daniel Han and team, 2023). To
control computation cost, we conduct asmall grid
searchon a 10% subsample of the external split
(Dext) and select the configuration with the best
validation loss. The selected configuration is then
trained on Dextfor one epoch. Our grid varies
only the learning rate, scheduler, and LoRA rank;
all other settings are fixed for stability and com-
parability. We always use 10% iterations for the
warm-up.
C Prompts
We provide the full prompts in Table 8, including
the system prompt and dataset-specific prompts.
Some datasets have features with anonymized
or symbolically named fields (e.g., V1–V28 ) that
carry limited semantic content for language mod-
els. For these datasets (CCFand IEEE-CIS), we
adoptschema-grounded prompts, which presentthe raw feature names and values directly. For
datasets with semantically interpretable attributes
(e.g.,CCFRAUD, PAYSIM), we employdescriptive
prompts, which paraphrase feature names into con-
cise natural language (e.g., “the transaction amount
is${TransactionAmt} ” rather than “ C1=... ”).
For retrieved exemplars, we format each exemplar
with the same schema-grounded or descriptive con-
vention as the target instance to avoid distribution
shift. At inference time, the model is instructed to
output a calibrated fraud score (Score: 1–5) and a
brief rationale.
D Inference Efficiency
We report additional measurements on (i) prompt
length (input tokens), (ii) output length (generated
tokens), and (iii) latency. We run inference with
vLLMand set the batch size to 1 to simulate an
online scenario. For each dataset-model config-
uration, we measure LLM generation latency (in
milliseconds) averaged over the evaluated queries.
We also report the average number of input tokens
14

Dimension Values
Learning rate {2e−4, 5e−5, 1e−5, 5e−6}
Scheduler {linear, cosine}
LoRA rank (r) {16, 32, 64}
Fixed settingsPrecision: bfloat16
Optimizer: AdamW (8-bit), weight decay= 10−2
LoRAα= 2×r, LoRA dropout= 0.05
Targeted modules: q_proj ,k_proj ,v_proj ,o_proj ,gate_proj ,
up_proj,down_proj
Seq length= 2048
Batching:batch size=8,gradient accumulation steps=2
Eval steps= 1000
Table 7: Tuning grid and fixed training settings for LLMs fine-tuning.
Role / DatasetPrompt
SystemYou are a helpful financial expert that can help analyze fraud. Use the fewest
reasoning steps needed to reach a correct answer. Please give a score of 1 to 5 for
the probability of fraud. You must includeScore:in your response. For example,
Score: 1means the lowest probability of fraud, andScore: 5means the highest
probability of fraud. Provide a brief explanation for your score.
RAGYou are given several similar historical cases with their ground truth labels.
Use them as guidance and then assess the current case.
Example 1: {example_transaction_prompt} It is a fraud/It is not a fraud.
. . .
Example N: {example_transaction_prompt} It is a fraud/It is not a fraud.
ccfThe client has only numerical input variables which are the result of a PCA
transformation: V10: {V10}, V14: {V14}, V4: {V4}, V12: {V12}, V11: {V11}, V17:
{V17}, V3: {V3}, V7: {V7}, V16: {V16}, V2: {V2}.
ccFraudThe client is a {gender}. the state number is {state}, the number of cards is
{cardholder}, the credit balance is {balance}, the number of transactions is {numTrans},
the number of international transactions is {numIntlTrans}, the credit limit is
{creditLine}.
IEEE-CISThe transaction has attributes: TransactionDT: {TransactionDT}, TransactionAmt:
{TransactionAmt}, card1: {card1}, C13: {C13}, card2: {card2}, C14: {C14}, addr1:
{addr1}, P_emaildomain: {P_emaildomain}, card6: {card6}, C1: {C1}.
PaySimThe transaction has: step is {step}, transaction type is {type}, amount is {amount},
originator is {nameOrig}, original balance before transaction is {oldbalanceOrg},
originator balance after transaction is {newbalanceOrig}, recipient is {nameDest},
recipient balance before transaction is {oldbalanceDest}, recipient balance after
transaction is {newbalanceDest}.
Table 8: Prompts by role and dataset. Features shown are post–feature-selection.
15

ModelCCF CCFRAUD PAYSIM IEEE-CIS
#Tok w/o #Tok w/ #Tok w/o #Tok w/ #Tok w/o #Tok w/ #Tok w/o #Tok w/
Qwen3 227.01 2925.23 161.28 1543.06 212.13 2610.96 210.87 2161.63
Gemma 3 224.01 2963.23 158.28 1581.06 206.71 2598.09 209.71 2225.74
GPT-OSS 262.00 2428.97 218.69 1518.26 243.70 2044.17 245.62 1756.99
Table 9: Average input-token counts (#Tok) with and without FinFRE-RAG.
ModelCCF CCFRAUD PAYSIM IEEE-CIS
Latency (ms) OutTok Latency (ms) OutTok Latency (ms) OutTok Latency (ms) OutTok
Qwen3-14B 18237.17 999.41 11057.95 540.26 9774.29 477.72 7748.80 378.37
Qwen3-14B + FinFRE-RAG 13838.09 621.09 11855.37 560.6 12904.11 601.37 10637.88 498.24
Qwen3-Next-80B 31740.25 3756.54 15765.56 1855.07 15495.72 1822.56 45496.92 5368.77
Qwen3-Next-80B + FinFRE-RAG 16472.47 1929.59 12608.28 1475.73 14964.63 1752.94 19905.43 2335.69
Gemma 3-12B 1415.67 73.47 1059.79 55.31 1452.68 75.55 1289.62 66.96
Gemma 3-12B + FinFRE-RAG 1300.47 49.42 1162.80 49.49 1387.43 55.76 1391.76 57.81
Gemma 3-27B 1592.12 101.91 1150.08 73.98 1383.32 88.56 1613.91 103.42
Gemma 3-27B + FinFRE-RAG 1390.03 70.65 1245.44 67.53 1583.24 87.61 1539.07 82.87
GPT-OSS-20B 1488.40 224.79 1638.03 248.97 1663.86 252.63 1569.56 235.72
GPT-OSS-20B + FinFRE-RAG 1600.38 221.47 2637.76 384.31 1398.36 193.96 2054.62 298.16
GPT-OSS-120B 1835.35 202.06 1928.25 213.78 2064.92 218.29 2384.15 249.35
GPT-OSS-120B + FinFRE-RAG 1828.55 188.2 1962.71 209.95 1877.77 197.47 2207.12 247.35
Table 10: Average latency and average output tokens (OutTok).
and output tokens. For reasoning-capable models,
we enable Thinking Mode for Qwen and set rea-
soning effort to Medium for GPT-OSS to reflect
practical usage.
Table 9 reports the average input-token count
with and without FinFRE-RAG. As expected,
FinFRE-RAG increases the prompt length substan-
tially because it injects retrieved in-context exem-
plars and their labels. Table 10 reports the aver-
age LLM generation latency together with the av-
erage number of output tokens. We count only
the time spent on LLM generation, excluding the
retrieval operation, as retrieval latency is highly
system-dependent and can vary substantially with
engineering choices (e.g., indexing, approximate
nearest-neighbor search, and caching). In some
settings, adding retrieved exemplars reduces aver-
age latency, as the presence of label-aware analogs
enables the model to converge more quickly during
generation and produce shorter responses. Con-
versely, for other configurations (e.g., GPT-OSS-
20B onCCFRAUD), FinFRE-RAG increases both
latency and output length, reflecting more detailed
rationales when the model is conditioned on multi-
ple exemplars.
16