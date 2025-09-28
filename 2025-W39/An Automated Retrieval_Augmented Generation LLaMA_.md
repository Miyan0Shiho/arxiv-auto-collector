# An Automated Retrieval-Augmented Generation LLaMA-4 109B-based System for Evaluating Radiotherapy Treatment Plans

**Authors**: Junjie Cui, Peilong Wang, Jason Holmes, Leshan Sun, Michael L. Hinni, Barbara A. Pockaj, Sujay A. Vora, Terence T. Sio, William W. Wong, Nathan Y. Yu, Steven E. Schild, Joshua R. Niska, Sameer R. Keole, Jean-Claude M. Rwigema, Samir H. Patel, Lisa A. McGee, Carlos A. Vargas, Wei Liu

**Published**: 2025-09-25 03:18:31

**PDF URL**: [http://arxiv.org/pdf/2509.20707v1](http://arxiv.org/pdf/2509.20707v1)

## Abstract
Purpose: To develop a retrieval-augmented generation (RAG) system powered by
LLaMA-4 109B for automated, protocol-aware, and interpretable evaluation of
radiotherapy treatment plans.
  Methods and Materials: We curated a multi-protocol dataset of 614
radiotherapy plans across four disease sites and constructed a knowledge base
containing normalized dose metrics and protocol-defined constraints. The RAG
system integrates three core modules: a retrieval engine optimized across five
SentenceTransformer backbones, a percentile prediction component based on
cohort similarity, and a clinical constraint checker. These tools are directed
by a large language model (LLM) using a multi-step prompt-driven reasoning
pipeline to produce concise, grounded evaluations.
  Results: Retrieval hyperparameters were optimized using Gaussian Process on a
scalarized loss function combining root mean squared error (RMSE), mean
absolute error (MAE), and clinically motivated accuracy thresholds. The best
configuration, based on all-MiniLM-L6-v2, achieved perfect nearest-neighbor
accuracy within a 5-percentile-point margin and a sub-2pt MAE. When tested
end-to-end, the RAG system achieved 100% agreement with the computed values by
standalone retrieval and constraint-checking modules on both percentile
estimates and constraint identification, confirming reliable execution of all
retrieval, prediction and checking steps.
  Conclusion: Our findings highlight the feasibility of combining structured
population-based scoring with modular tool-augmented reasoning for transparent,
scalable plan evaluation in radiation therapy. The system offers traceable
outputs, minimizes hallucination, and demonstrates robustness across protocols.
Future directions include clinician-led validation, and improved domain-adapted
retrieval models to enhance real-world integration.

## Full Text


<!-- PDF content starts -->

An Automated Retrieval-Augmented Generation LLaMA-4
109B-based System for Evaluating Radiotherapy Treatment
Plans
Junjie Cui1, Peilong Wang, PhD1, Jason Holmes, PhD1, Leshan Sun, PhD1, Michael
L. Hinni, MD2, Barbara A. Pockaj, MD3, Sujay A. Vora, MD1, Terence T. Sio, MD,
MS1, William W. Wong, MD1, Nathan Y. Yu, MD1, Steven E. Schild, MD1, Joshua
R. Niska, MD1, Sameer R. Keole, MD1, Jean-Claude M. Rwigema, MD1, Samir H.
Patel, MD1, Lisa A. McGee, MD1, Carlos A. Vargas, MD1, and Wei Liu, PhD∗1
1Department of Radiation Oncology, Mayo Clinic Arizona, Phoenix, AZ 85054
2Department of Otolaryngology, Mayo Clinic Arizona, Phoenix, AZ 85054
3Department of General Surgery, Mayo Clinic Arizona, Phoenix, AZ 85054
September 26, 2025
Abstract
Purpose
To develop a retrieval-augmented generation (RAG) system powered by LLaMA-4 109B for
automated, protocol-aware, and interpretable evaluation of radiotherapy treatment plans.
Methods and Materials
We curated a multi-protocol dataset of 614 radiotherapy plans across four disease sites and con-
structed a knowledge base containing normalized dose metrics and protocol-defined constraints.
The RAG system integrates three core modules: a retrieval engine optimized across five Sen-
tenceTransformer backbones, a percentile prediction component based on cohort similarity, and
a clinical constraint checker. These tools are directed by a large language model (LLM) using a
multi-step prompt-driven reasoning pipeline to produce concise, grounded evaluations.
Results
Retrieval hyperparameters were optimized using Gaussian Process on a scalarized loss function
combining root mean squared error (RMSE), mean absolute error (MAE), and clinically moti-
vated accuracy thresholds. The best configuration, based onall-MiniLM-L6-v2, achieved perfect
nearest-neighbor accuracy within a 5-percentile-point margin and a sub-2pt MAE. When tested
end-to-end, the RAG system achieved 100% agreement with the computed values by standalone re-
trieval and constraint-checking modules on both percentile estimates and constraint identification,
confirming reliable execution of all retrieval, prediction and checking steps.
Conclusion
Our findings highlight the feasibility of combining structured population-based scoring with
modular tool-augmented reasoning for transparent, scalable plan evaluation in radiation ther-
apy. The system offers traceable outputs, minimizes hallucination, and demonstrates robustness
across protocols. Future directions include clinician-led validation, and improved domain-adapted
retrieval models to enhance real-world integration.
1 Introduction
Radiotherapy treatment planning aims to generate a treatment plan that delivers a therapeutic dose
to the tumor while minimizing radiation exposure to surrounding healthy tissues [1, 2, 3, 4, 5, 6].
An essential step in this workflow is the plan evaluation, in which the quality of the proposed dose
∗Corresponding author
1arXiv:2509.20707v1  [cs.AI]  25 Sep 2025

distribution is assessed. Plan quality refers to the clinical suitability of the dose distribution that a
treatment plan can reasonably achieve with optimal coverage of the target by the prescribed dose while
minimizing dose to normal tissues [7]. This is typically evaluated through a quantitative analysis of
the dose distribution produced by the treatment planning system (TPS) [8] or by manual peer review
[9]. However, this approach is time-consuming and often involves subjective judgments that may vary
between clinicians [9, 10].
Given the limitations of manual evaluation, extensive efforts [7, 11, 12, 13, 14, 15, 16, 17, 16, 18]
have been made to develop objective and efficient approaches for radiotherapy plan evaluation using
statistical and mathematical frameworks. A common strategy is population-based scoring, in which
plan quality is evaluated relative to a cohort distribution. For example, Leone et al.[9] proposed a
geometric mean–based scoring system that normalizes each dose–volume histogram (DVH) indices by
its clinical limit and ranks treatment plans by their percentile within a protocol-specific cohort. This
method demonstrated strong correlation with physician Likert ratings and provided a cohort-aware
framework for quality assessment. Other approaches have focused on statistical modeling of DVH
distributions. Wahl et al.[19] introduced a probabilistic model that analytically estimates expected
DVHs andα-DVHs under delivery uncertainties, enabling robust plan evaluation without the need
for Monte Carlo simulation. In a complementary direction, population-derived visual dashboards
have been developed to benchmark DVH indices against historical distributions, offering intuitive
and interactive feedback via statistical overlays [20]. Meanwhile, mathematical optimization methods
have also been employed to assess and improve plan quality directly. Engberg et al.[21] formulated a
convex optimization framework using mean-tail-dose as a risk measure, allowing more precise control
over DVH tail behavior. Zhang et al.[22] extended this line of work by treating DVH indices as
differentiable functionals, enabling direct optimization of clinically relevant constraints. To facilitate
the large-scale application of such frameworks, Pyakuryal et al. [23] developed the Histogram Analysis
in Radiation Therapy (HART) tool, which automates the extraction and statistical analysis of DVH
indices, significantly enhancing the reproducibility and efficiency of plan evaluation. Collectively, these
works demonstrate that statistical and mathematical methods can provide rigorous, interpretable,
and clinically grounded strategies for automated plan evaluation. However, such approaches often
require hand-engineered metrics, are limited to predefined protocols, and may not generalize well
across institutions with different naming conventions or contouring practices [7, 13]. Furthermore, their
outputs tend to be static and lack the interactive, explanatory capabilities needed to support dynamic
clinical workflows or accommodate evolving clinical guidelines. These limitations have motivated
interests in more flexible, learning-based systems, particularly those powered by large language models
(LLMs), which can reason over heterogeneous inputs, provide interpretable summaries, and integrate
auxiliary tools to support complex decision-making [24, 25].
Recent advances in LLMs, a class of deep learning models trained on massive corpora of texts, have
opened new opportunities for automating complex clinical workflows. LLMs are capable of perform-
ing natural language understanding, multi-turn reasoning, and code execution, and can interface with
external tools to support decision-making tasks [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]. These capabil-
ities arise from their ability to learn high-dimensional representations of language and generalize across
diverse inputs without task-specific supervision [37]. As a result, LLMs are increasingly being explored
for medical applications that require interpreting clinical text, synthesizing structured and unstruc-
tured data, and generating human-readable summaries [38, 39]. In particular, retrieval-augmented
generation (RAG) systems combine the generative capabilities of LLMs with external knowledge re-
trieval mechanisms, allowing the model to incorporate relevant, case-specific contexts into its reasoning
process [40]. A typical RAG workflow is illustrated in Fig. 1, where the user query triggers a retrieval
step that fetches relevant documents, which are then passed to the LLM for grounded response gener-
ation. This paradigm has shown promise in domains like clinical decision support and radiology report
generation, and is now beginning to impact radiotherapy planning [41]. For instance, a recent feasi-
bility study by Wang et al.[10] demonstrated that GPT-4-based agents could partially automate the
radiotherapy treatment planning process, including dose prescription, structure prioritization, and plan
evaluation, using chain-of-thought prompting and self-verification. Similarly, Liu et al.[42] explored
a multi-agent LLM system for guiding the full radiotherapy pipeline, including image interpretation
and treatment planning, through GPT-4 Vision and command-executing submodules. Although these
systems aim to automate the entire planning process, their results highlight the importance of accurate
and interpretable plan evaluation as a critical subtask. In parallel, domain-specific applications of RAG
2

Figure 1: Overview of the retrieval-augmented generation (RAG) workflow. A user query is processed
by a retriever to fetch relevant context from a knowledge base, which is then fed into an LLM for
grounded generation.
systems have emerged, in which LLMs are guided by retrieved dose-volume metrics and clinical pro-
tocols to assess new treatment plans against historical cohorts [39, 43]. These approaches represent a
promising direction for building clinically informed, interpretable, and adaptable tools for radiotherapy
plan evaluation.
To address the need for accurate, interpretable and scalable plan evaluation, we introduce an auto-
mated RAG system powered by LLaMA-4 109B for radiotherapy quality assessment. Our framework
integrates (1) a scoring module that computes normalized dose metrics and population-based per-
centiles, (2) a retrieval module that identifies similar historical plans based on numerical and textual
features, and (3) a constraint-checking tool that flags clinical violations using protocol-defined thresh-
olds. These components are leveraged through an LLM-driven reasoning process that issues explicit
tool calls to retrieve context, perform checks, and synthesize a structured, protocol-aware summary
of plan quality. Unlike prior end-to-end systems, our method separates data, logic, and generation,
reducing hallucinations while supporting traceability and flexible protocol integration. The output in-
cludes a quantitative plan score and a list of failed constraints, enabling transparent decision support
aligned with clinical practice. We curate a multi-protocol knowledge base from 614 historical plans
and demonstrate the system’s accuracy and reliability through evaluation.
2 Method
All research activities of this study were conducted under institutional review board (IRB) approval.
The IRB number associated with this study is 24-010322: “Application of Large Language Models
(LLMs) to Enhance Efficiencies of Clinical and Research Tasks in Radiation Oncology”.
2.1 Dataset and Clinical Protocols
We curated a dataset of 614 radiotherapy treatment plans across 463 patients, covering four disease
sites: head and neck (3 protocols, 175 plans), prostate (3 protocols, 264 plans), breast (2 protocols, 136
plans) and lung (1 protocol, 48 plans). All cases were drawn from clinical trials conducted at the Mayo
Clinic Arizona. We focused on clinical trial data because such plans are consistently annotated and
3

strictly follow protocol-defined dose constraints for target regions and organs-at-risk (OARs), making
them ideal for training and evaluating a protocol-aware system.
Initial identification of eligible plans was facilitated using RadOnc-GPT [44], a domain-specialized
large language model capable of retrieving unstructured oncology information from electronic records
and assisting in clinical cohort discovery. Each plan included a protocol-defined set of mandatory
DVH endpoints, which were used to guide evaluation. A full breakdown of the protocols and the
corresponding DVH endpoints is provided in Table 1.
Protocol Name Required DVH Endpoints
Lung 1 Esophagus D33%, D67%, D100%; Liver D100%,
D50%; Cord Max; Heart D33%, D67%, D100%, Mean,
V50Gy; Skin Max; Lung Total V20Gy
Head & Neck 1 Cord Max; Brain Stem Max; Lips Mean; Oral Cavity
Mean; Parotid Mean; Esophagus Mean; Submandibu-
lar Mean; Larynx Mean
Head & Neck 2 Brain Stem Max; Parotid/Submandibular/Pharyngeal/
Larynx/Oral Cavity Mean; Mandible Max; Brachial
Plexus Max
Head & Neck 3 Cord D0.03cc; Brain Stem D0.03cc; Lips/Oral
Cavity/Parotid/Submandibular/Larynx Mean; OAR
Pharynx Mean, V15%, V33%; Cervical Esophagus
Mean
Breast 1 Heart D0.01cc, Mean; Breast Skin D0.01cc; Lung
Ipsilateral/Contralateral V40%; Esophagus D0.01cc;
Brachial Plexus D0.01cc
Prostate 1 Rectum V65%, V90%, D0.03cc; Bladder V90%,
D0.03cc; Femoral Heads (L/R) V50cc
Prostate 2 Rectum V30Gy; Bladder V30/33cc; Femoral Heads
V40cc; Small/Large Bowel Max; Lung Total V30Gy
Prostate 3 (Multiple arms) Rectum/Bladder V40–70Gy, Dmax;
Femoral Heads V40cc or Max
Breast 2 (Photon/Proton arms) Heart Max, Mean; Breast Skin
Max; Lung Ipsilateral/Contralateral V50%
Table 1: Summary of clinical protocols used in this study, including disease sites, number of patients,
number of treatment plans, and the protocol-defined DVH indices used for evaluation.
Patient demographics for the full cohort are summarized as follows: the mean age was 64.8 years
(range: 29–91; standard deviation (SD): 10.4). The majority of patients identified as White (91.4%),
with the remainder distributed across Black or African American (2.2%), Native American (2.6%),
Asian (2.6%), and other or undisclosed categories (1.3%).
To preprocess the data, we first extracted the protocol-defined dose volume constraints directly from
official protocol documents housed in the institution’s clinical trial database. Plans were organized and
saved into separate text files based on their protocol assignments, ensuring that each file contained only
the relevant dose metrics and identifiers for downstream evaluation. We then parsed each treatment
plan to extract the DVH indices required by its associated protocol. This structured preprocessing
step enabled streamlined normalization and scoring in subsequent stages of system development.
2.2 Knowledge Base–Enabled Plan Scoring
Our plan evaluation framework is inspired by previous work on population-based scoring by Leone et
al. [9], which showed that aggregating protocol-defined DVH indices can yield clinically interpretable
assessments of plan quality. In our system, each clinical protocol specifies a set of DVH indices that
must be evaluated for specific OARs or targets. These indices may include metrics such as D33% (the
minimum dose received by the hottest 33% of a structure’s volume) and V50Gy (the percentage of
a structure’s volume receiving at least 50 Gray). These metrics reflect both high-dose regions and
volumetric coverage, and are widely used to ensure both tumor control and healthy tissue sparing.
For each treatment plan, we extract the raw values of all protocol-required DVH indices. To
allow fair aggregation across metrics with different units and scales, each raw value is then normalized
by dividing it by its corresponding clinical constraint (e.g., a dose threshold or volume limit), and
4

multiplied by 100 to express it as a percentage:
normalized score i=raw value i
constraint limit i×100 +ε.
Here,ε= 10−6is a small constant added to prevent zero values and numerical instability.
To obtain a single summary score for the entire plan, we compute the geometric mean of the
normalized values across all DVH indices:
gmscore = nY
i=1normalized score i!1/n
.
This approach ensures that no single large violation is masked by low values in other metrics, as the
geometric mean is more sensitive to outliers than an arithmetic mean. It also naturally balances all
constraints without requiring hand-tuned weights.
Finally, the geometric mean score is mapped to a percentile rank within the historical cohort of
plans for the same protocol. This percentile serves as the final plan score and enables direct comparison
across different plans operating under shared clinical standards. A schematic overview of the full plan
scoring pipeline, from raw DVH indices to percentile-based plan scores, is illustrated in Fig. 2.
Figure 2: Overview of the plan scoring pipeline. For each treatment plan, protocol-defined DVH indices
(e.g., D33%, V50Gy) are extracted and normalized against their corresponding protocol constraints.
These normalized values are aggregated using the geometric mean to compute a single value, which is
then mapped to a percentile rank within the protocol-matched cohort. The percentile rank is the final
score.
To support retrieval and reasoning, each scored plan is stored as a JavaScript Object Notation
(JSON) entry containing its protocol name, raw DVH indices, normalized metrics, geometric mean,
and percentile rank. To enable evaluation of retrieval and generalization, the plans were partitioned
by protocol: 90% were used to build the knowledge base (579 plans) for our RAG system, while the
remaining 10% were held out as a testing dataset (62 plans). For testing plans, percentile scores
were hidden at the inference time, allowing the RAG system to generate plan-level assessments using
retrieved examples and protocol-based reasoning.
2.3 Retrieval and Prediction
To predict the clinical quality of a new radiotherapy treatment plan, we designed a retrieval-based
system that identifies similar historical plans from the knowledge base and uses their percentile scores
to estimate the quality of the input plan. For each protocol, we constructed three similarity indexes
over the historical plans: (1) a text-based index encoding a natural language representation of each
plan’s DVH indices using sentence embeddings; (2) a normalized-metric index built from numerical
vectors of protocol-defined metrics, each normalized as a percentage of its clinical limit; and (3) a
raw-metric index constructed using the unnormalized DVH index values.
At the inference time, given a new test plan, we first retrieve the topkplans with the most similar
geometric mean scores from the same protocol. These initial candidates are then re-ranked using a
weighted similarity score that combines three components: semantic similarity between the textual
descriptions of metric values, vector similarity of normalized metric profiles, and vector similarity of
raw metric values. The similarity weights are tunable, allowing flexible control over the influence of
each component. For each test plan, we report the nearest neighbor’s percentile score, as well as a
5

weighted average and weighted median of the retrieved neighbors’ percentiles, which together form the
final predicted quality estimates. This multi-view retrieval strategy allows the system to generalize
across protocols and capture both numerical and semantic patterns in historical plan data.
2.4 RAG System for Plan Evaluation
We developed an RAG system powered by LLaMA 4 109B to support the interpretation and evaluation
of radiotherapy treatment plans. The system integrates structured dose-volume data with protocol
knowledge to produce concise and clinically relevant assessments. An overview of this workflow is
illustrated in Fig. 3.
Figure 3: Overview of our RAG system workflow for radiotherapy plan evaluation. Given a new
treatment plan, the system first computes DVH indices normalized to the protocol-defined constraints
and the geometric mean of the normalized dose metrics. The geometric mean, the normalized and
raw DVH indices are used to query a retrieval module, which retrieves similar historical plans from
a protocol-matched knowledge base, and makes predictions on the test plan quality based on the
retrieved historical plans. In parallel, a constraint-checking module identifies violations based on the
protocol constraints. The retrieved plans and constraint results are then passed to LLaMA 4 109B,
which synthesizes a concise and interpretable summary describing the percentile-based plan quality
and any failed constraints.
Upon receiving a new treatment plan, the system first computes dose metrics normalized to the
protocol-defined constraints, along with a summary score reflecting overall plan quality. This struc-
tured representation serves as input to the RAG system. The LLM is equipped with two auxiliary
tools: a retrieval module that identifies similar historical plans based on both textual and numerical
features, and a constraint-checking module that flags violations of protocol constraints.
The system operates in a multi-step, tool-augmented reasoning process. First, the LLM queries the
retrieval module to obtain the percentile estimates for the input plan, including its similarity-based
nearest neighbor, weighted average and median within the protocol-matched cohort. Next, it invokes
the constraint-checking tool to determine which dose-volume metrics exceed clinical constraints. With
this contextual information, the LLM then generates a brief and interpretable summary describing the
plan’s percentile-based standing and any constraint violations.
6

This design allows the LLM to produce grounded, protocol-aware evaluations while minimizing
hallucinations. By combining structured retrieval, constraint analysis, and generative reasoning in a
unified framework, the RAG system enables interpretable and scalable plan evaluation across diverse
clinical protocols.
2.5 Experimental Design
2.5.1 Experiment 1: Retrieval Accuracy and Ranking Quality
The goal of our first experiment was to identify the optimal retrieval strategy for the RAG system
by evaluating multiple SentenceTransformer backbones and tuning the associated retrieval related
configurations. SentenceTransformer contains a library of pretrained language models used to convert
text (e.g., structure names) into dense numerical vectors (embeddings) that capture semantic similarity.
These embeddings allow for comparisons between textual features based on cosine similarity or other
distance metrics in the latent space. The backbone architecture affects how well these embeddings
capture clinical semantics, making it a critical design choice for retrieval tasks involving structured
radiotherapy data.
Each SentenceTransformer backbone was evaluated under an independently optimized retrieval
configuration. Specifically, we tuned the similarity weighting coefficients for the three retrieval com-
ponents: the weighting coefficients for textual similarity (α), normalized metric distance (β norm), and
raw metric distance (β raw). These weights control the contribution of each feature type when com-
puting similarity between the query and candidate plans. In addition, we tuned the retrieval depthk,
which specifies the number of nearest neighbors retrieved to form the contextual cohort for percentile
estimation. We restrictedkto the range of 3 to 10 for methodological and practical reasons. Setting
k= 1 would bypass any aggregation and rely solely on the nearest neighbor, making it incompatible
with our weighted mean and median calculations. A cohort size ofk= 2 similarly lacks statistical
robustness for percentile estimation. On the other end, increasingkbeyond 10 risks incorporating
weakly similar plans due to the nature of percentile-based scoring and the limited size of our dataset.
To optimize these hyperparameters, we employed Gaussian Process (GP) minimization, a sample-
efficient black-box optimization method. In our setting, each evaluation of the retrieval configuration
requires computing multiple prediction metrics over all test plans, which is computationally expensive.
GP optimization is particularly suited to such scenarios: rather than exhaustively testing all combi-
nations of weights and retrieval depths, it builds a probabilistic model that estimates how changes
in the hyperparameters affect performance. It then intelligently selects the next configuration to try
by balancing the exploration of new regions of the parameter space with the exploitation of known
high-performing regions. This approach allowed us to efficiently search for the best combination of
similarity weights (α,β norm,βraw) and retrieval depthkfor each SentenceTransformer backbone, min-
imizing the need for brute-force tuning while ensuring that the selected configurations yield clinically
accurate percentile predictions.
The objective for optimization was a scalarized loss function that integrates four clinically moti-
vated evaluation metrics: (1) root mean squared error (RMSE AVG) between the predicted and true
percentiles using weighted average acrosskretrieved plans; (2) mean absolute error (MAE NN) between
the predicted and true percentiles using the single nearest neighbor; (3) percentage of cases where the
1-nearest-neighbor prediction is within 5 percentile points of the true percentile score (%≤5pt NN);
and (4) percentage of cases where the weighted average prediction is within 10 percentile points
(%≤10pt AVG). The loss function is defined as the sum of RMSE, MAE, and the complement of
the two accuracy metrics (normalized to the [0,1] scale):
L= RMSE AVG+ MAE NN+100−%≤5ptNN
100+100−%≤10ptAVG
100
This penalizes both large errors and clinically unacceptable predictions. This optimization process was
repeated independently for each model to ensure a fair comparison under individually tuned settings.
Notably, this tuning process did not involve any fine-tuning of model weights for the Sentence-
Transformer or LLM. Instead, our approach treats these components as frozen modules and focuses
on tuning the retrieval logic, similarity weights, and the system prompt. This modularity supports
generalizability, ease of deployment, and flexible integration into clinical environments.
7

For the best-performing retrieval model, we report a suite of evaluation metrics to assess both the
accuracy and clinical relevance of percentile predictions. These include mean absolute error (MAE),
root mean squared error (RMSE), Pearson correlation coefficient (r), Spearman rank correlation co-
efficient (ρ), coefficient of determination (R2), and the percentage of predictions within 5 and 10
percentile points of the true percentile score. MAE captures the average absolute difference between
the predicted and actual percentiles, providing an intuitive sense of the typical deviation encountered
in practice. RMSE penalizes larger errors more heavily and thus helps highlight occasional outlier
mismatches that may have greater clinical impact. Pearsonrquantifies the linear correlation between
the predicted and true percentiles, offering insights on how well the model preserves the overall trend
in plan quality. Meanwhile, Spearmanρassesses monotonicity and rank consistency, which is espe-
cially useful in clinical decision-making scenarios where plans are selected or triaged based on relative
performance rather than precise numeric scores. The value ofR2reflects the proportion of variance
in the true percentile score explained by the model, providing a high-level measure of its predictive
power. Finally, we report the percentage of test cases, where the predicted percentile falls within 5 or
10 points of the true value: two thresholds that reflect common clinical tolerances for stratifying plan
quality. These thresholds offer a pragmatic view of the system’s ability to deliver clinically acceptable
estimates under real-world scenarios. Collectively, these metrics offer a multidimensional perspective
on retrieval performance, encompassing both statistical rigor and clinical interpretability, and support
our selection of the best-performing SentenceTransformer backbone model as the default retriever for
our RAG-based plan evaluation system.
2.5.2 Experiment 2: Evaluation Quality of the RAG System
The second experiment aimed to evaluate whether the RAG system, when integrated with LLaMA
4 109B, could produce outputs consistent with those obtained from the individual scoring, retrieval
and constraint-checking modules. Prior to the full-scale evaluation, we refined the system prompts
to ensure that the LLM followed the intended sequence of the tool calls and returned outputs in
the expected format. Once the prompts were finalized, we executed the complete pipeline for each
test plan, including plan scoring, retrieval using the optimal configuration from Experiment 1, and
LLM-based reasoning in a batch processing setting. The generated summaries were then collected for
evaluation.
To assess the system accuracy, we compared the percentile estimates reported by the LLM (nearest
neighbor, weighted average, and weighted median) with the reference values computed directly from
the underlying retrieval and scoring components. Consistency was determined by parsing the model
responses and automatically identifying discrepancies. In parallel, we evaluated the integration of the
system with the constraint checking module by verifying that the LLM correctly invoked the relevant
tool call and that the returned set of the violated constraints matched those identified.
This experiment confirms that the RAG system reliably executes the intended sequence of oper-
ations (retrieval, prediction, and constraint checking) and generates outputs aligned with expected
outputs in real-world clinical scenarios.
3 Results
3.1 Plan Retrieval Accuracy
We evaluated five SentenceTransformer backbone models to identify the optimal backbone model for
retrieving clinically similar plans in our RAG system. Each model differs in architecture, size, and
pretraining corpus. Therefore, finding the model that fits our system is crucial. For this study, we
selected models spanning a range of tradeoffs between performance and efficiency:all-MiniLM-L6-v2
andall-mpnet-base-v2are smaller and faster, making them more suitable for real-time applications;
stsb-roberta-largeandall-distilroberta-v1are slower but potentially more accurate because
they process languages in greater detail; andmsmarco-distilbert-base-tas-bis optimized for re-
trieval tasks. All models were used without fine-tuning.
To ensure fair comparison, we independently optimized each model’s retrieval configuration using
the method described in Experiment 1. This involved selecting the optimal similarity weights and
retrieval depthkusing GP based minimization, targeting a combined loss function over four clinically
motivated prediction metrics.
8

Table 2 summarizes the best configurations for each model along with their retrieval accuracy.
Despite the technical differences among backbone models, all achieved perfect accuracy in retrieving
a nearest neighbor within 5 percentile points of the true plan score. However,all-MiniLM-L6-v2
achieved the lowest overall scalarized loss (5.53), reflecting slightly better prediction accuracy and
robustness. In practice, this model offers a compelling balance between performance and computational
cost, requiring only a fraction of the memory and runtime of larger models such as RoBERTa-large.
This makes it a strong candidate for integration into clinical systems. We therefore selectedMiniLM
for our default SentenceTransformer backbone.
Model k α βnorm βraw RMSE AVG MAE NN %≤5pt NN %≤10pt AVG Loss
all-mpnet-base-v2 40.000080 0.999793 0.000127 3.777993 1.770161 100.0 98.387097 5.564283
all-MiniLM-L6-v2 40.004313 0.983081 0.012606 3.777999 1.738065 100.0 98.387097 5.532193
all-distilroberta-v1 40.000045 0.999909 0.000045 3.777993 1.770161 100.0 98.387097 5.564284
msmarco-distilbert-base-tas-b 40.000045 0.999909 0.000045 3.777997 1.770161 100.0 98.387097 5.564287
stsb-roberta-large 40.000076 0.999879 0.000045 3.778000 1.738065 100.0 98.387097 5.532194
Table 2: Optimized retrieval configurations and summary metrics across the SentenceTransformer
backbone models. The best-performing model isall-MiniLM-L6-v2.
To further evaluate the selected retrieval model (MiniLM), we examined its prediction performance
using three common aggregation strategies: nearest neighbor, weighted average, and weighted median.
As shown in Table 3, all models achieved high accuracy. The nearest neighbor approach delivered the
lowest mean absolute error (MAE = 1.74) and perfect agreement (100%) within 5 percentile points of
the true percentile score, indicating strong consistency for individual predictions. While the weighted
average method had slightly higher MAE and RMSE values, it achieved the best ranking consistency, as
reflected by the highest Spearman correlation (ρ= 0.9902). This means that the predicted percentiles
followed the same order as the true percentiles more faithfully, which is especially important when
treatment plans are selected or prioritized based on the relative ranking.
Method Pearsonr Spearmanρ MAE RMSE R2%≤5pt %≤10pt
Nearest Neighbor 0.9971 0.9959 1.7381 1.9013 0.9942 100.0 100.0
Weighted Average 0.9895 0.9902 2.1361 3.7780 0.9769 91.94 98.39
Weighted Median 0.9944 0.9943 1.9986 2.7962 0.9874 90.32 100.0
Table 3: Prediction performance of the selected retrieval model (all-MiniLM-L6-v2) across multiple
aggregation strategies.
Overall, these results confirm that our retrieval-based prediction pipeline can identify high-quality
reference plans with remarkable accuracy, often within 1–2 percentile points of the true percentile
scores. This demonstrates the effectiveness of semantically enhanced plan retrieval for cohort-aware,
protocol-specific evaluation.
3.2 LLM-Based Plan Evaluation Accuracy
In the Experiment 2, we evaluated whether the full RAG system integrating retrieval, percentile
prediction, and constraint checking could produce outputs consistent with values obtained from the
individual modules. For each test plan, the system was run end-to-end in the batch mode.
We verified that the LLM executed the expected sequence of operations. Specifically, it retrieved
relevant plans using the optimized configuration from Experiment 1, generated percentile predictions
based on the retrieved values, and subsequently triggered the constraint-checking tool to identify any
violations of clinical constraints. These tool calls were implicitly embedded in the LLM’s response,
and their outcomes were parsed and compared against the values computed by the standalone retrieval
and constraint-checking modules.
For all evaluated plans, the LLM-generated summaries yielded exact matches with the reference
values, achieving 100% agreement with both predicted percentiles (nearest neighbor, weighted average,
and weighted median) and clinical constraint violations. This demonstrates that the system not only
performs correct retrieval and scoring but also integrates tool outputs into a coherent and clinically
interpretable narrative. Fig. 4 illustrates one such example. On the left, we show the full summary
generated by the LLM in response to a test plan. On the right, we display the individual components
parsed from the output: the three predicted percentile estimates and the reported constraint violations
9

for the heart structure. All values are in perfect agreement with the computed values from individual
modules, highlighting the reliability and transparency of our RAG system in mimicking the behavior
of modular pipelines while producing human-readable summaries.
Figure 4: An example demonstrating consistency between the LLM-generated summary (left) and
the parsed outputs (right), which include the predicted percentile scores (nearest neighbor, weighted
average, and weighted median) and a flagged constraint violation (Heart mean dose (Gy)). All outputs
match the values produced by the underlying retrieval and constraint-checking modules, confirming
faithful execution and integration within the RAG pipeline.
4 Discussion
This work presents an LLaMA 4 109B-based RAG system for automating radiotherapy treatment
plan evaluation. We found that our system accurately retrieves clinically similar plans, predicts per-
centile values with high fidelity, and identifies violated constraints in full agreement with expected
outputs. Notably, the LLM demonstrated perfect consistency with independently verified retrieval
and constraint-checking tools, achieving 100% accuracy across all test cases.
4.1 Key Observations from Retrieval Tuning
The results of Experiment 1 provide key insights into how percentile prediction accuracy can be
optimized for radiotherapy plan evaluation. Although all five SentenceTransformer backbone models
performed similarly on standard error-based metrics (e.g., RMSE, MAE), our optimized retrieval
configurations consistently prioritized structured and numerical DVH indices over textual metadata.
Specifically, the weight assigned to textual similarity between structure names was negligible across
models, while the majority of the retrieval signal came from the normalized differences in dose metrics.
This suggests that in our dataset, plan identifiers and structure labels did not offer sufficient variation
or semantic richness to meaningfully guide retrieval. In contrast, numerical features directly reflected
inter-patient variation in treatment delivery and were more effective for identifying clinically similar
plans. An important takeaway is the strength of the nearest neighbor method for generating percentile
estimates. This simple approach not only achieved the lowest prediction error, but also demonstrated
perfect agreement within±5 and±10 percentile thresholds. These results suggest that, in contexts
where a library of similar plans is available, retrieving the most relevant prior case can yield clinically
acceptable and highly interpretable benchmarks without requiring complex aggregation or averaging.
10

Together, these findings reinforce the value of the structured dose information in radiotherapy plan
analytics and suggest that advanced language embeddings may offer limited additional benefits when
dose metrics are already well-curated and standardized.
4.2 Interpretability and Modular System Design
Our results illustrate that a retrieval-augmented and modular approach yields both clinically accurate
and highly interpretable treatment plan evaluations. By grounding percentile predictions in struc-
tured and numerical similarity rather than relying on opaque embedding-based reasoning, our system
avoids the pitfalls of “black-box” AI models, such as unpredictable behaviors and clinician mistrust,
while still leveraging LLMs in a controlled, tool-mediated manner. This aligns with broader trends
emphasizing the necessity of transparency and explainability in medical AI systems, especially when
high-stakes decisions are involved [45, 46]. In contrast to many end-to-end learning systems, our design
enables traceability: each prediction can be traced back to the specific reference plans and associated
constraints. This model of “glass-box” decision support resonates with recommendations from the
clinical AI literature, which emphasize auditability and explainable outputs to promote safety and
clinician acceptance [45, 47]. The dominant role of numerical DVH-based similarity—versus semantic
embeddings—in our retrieval configurations suggests that structured clinical features are often more
informative for risk estimation than free-text or metadata encoding. This observation echoes findings
from case-based reasoning frameworks in healthcare, where feature-based nearest neighbor retrieval
has shown strong performance in diagnosis and treatment recommendation tasks [48, 47]. Finally, our
system’s separation of retrieval, constraint logic, and LLM-based reasoning naturally supports iterative
refinements and human-in-the-loop workflows. This design not only enhances safety and adaptabil-
ity but also better aligns with regulatory and institutional requirements governing AI in healthcare
deployment [49, 50].
4.3 Limitations and Methodological Considerations
While our system demonstrates high accuracy and robustness, several limitations must be acknowl-
edged. Most notably, the dataset size remains constrained due to limited patient enrollment in clinical
trials and the inherent challenges of accessing radiotherapy plans with appropriate patient consent.
This restricts the diversity of case presentations and may limit generalizability to rare or highly individ-
ualized treatment contexts. In addition, although we tested a range of SentenceTransformer backbone
models, all models exhibited minimal reliance on sentence embeddings, favoring numerical dose metrics
instead. This suggests that current general-purpose LLMs may not fully capture radiotherapy-specific
semantic nuances, and we did not explore more targeted alternatives such as domain-adapted encoders
or LLMs fine-tuned based on medical corpora. This choice was driven by time and resource constraints.
In future work, we plan to investigate the use of clinical-domain encoders (e.g., BioBERT, PubMed-
BERT) or fine-tuning strategies tailored to radiotherapy-specific domain knowledge. We hypothesize
that such models may improve performance in settings where semantic structure names play a larger
role or where DVH index information alone is insufficient for disambiguation. The scalarized loss func-
tion used to fine tune retrieval hyperparameters played a vital role in consolidating multiple evaluation
metrics into a single, reproducible optimization target. However, because there is no formally defined
similarity function for percentile estimation in this domain, the extent to which this loss captures
clinical relevance or predictive confidence remains uncertain. To address this, we plan to benchmark
the scalarized loss function against downstream clinical outcomes and conduct calibration studies to
evaluate how well its predictions align with expert interpretation.
While our normalized DVH metrics are derived using protocol-defined endpoints, we acknowledge
that differences in calculation methods, such as dose grid resolution, interpolation, or ROI definitions,
may lead to discrepancies compared to values observed in clinical systems. These differences could
affect constraint evaluations for certain plans. Although we have aimed for consistent implementation
across all records, further harmonization with clinical systems and validation against source planning
data may help reduce such inconsistencies.
11

4.4 Future Work and Clinical Validation
Moving forward, we aim to extend validation beyond retrospective consistency checks by conducting
prospective user studies involving radiation oncologists. These studies will assess whether system-
generated plan evaluations align with clinical judgment and improve decision-making workflows. In
particular, we see strong potential for our system to support adaptive treatment planning scenarios,
where clinicians must select or revise plans in near real-time while the patient is on the treatment table.
Since our retrieval-augmented framework can rapidly surface clinically similar reference plans and pro-
vide interpretable constraint-based summaries, it may offer decision support during tight time windows
when adaptive planning is necessary. Future evaluations will explore whether the system’s latency,
interpretability, and accuracy are sufficient to assist in such time-sensitive clinical settings. Clinician
feedback will also inform the development of more principled retrieval objectives and may motivate
adjustments to feature weighting schemes or similarity metrics. In parallel, we will explore automated
prompt engineering methods, such as reinforcement learning with human feedback (RLHF) or con-
trastive prompt optimization, to enhance robustness under diverse inputs. Finally, future versions of
our system may incorporate richer clinical contexts, such as physician notes, anatomical imaging data,
or planning intent, to support more holistic and personalized radiotherapy treatment plan evaluation.
5 Conclusion
We presented a retrieval-augmented generation LLaMA 4 109B system for evaluating radiotherapy
treatment plans in a protocol-aware and interpretable manner. By combining structured population-
based scoring, numerical and semantic retrieval, constraint checking, and LLM–driven summarization,
our system offers a transparent, modular framework for clinical radiotherapy plan assessment.
Our experiments demonstrate that the system achieves high accuracy in both retrieval-based per-
centile prediction and constraint violation identification, with perfect agreement with computed values
from individual modules in all evaluated cases. Notably, we found that normalized dose metrics play
a dominant role in determining plan similarity with minimal contribution from sentence embeddings,
an insight that underscores the value of structured clinical features over free-text representations in
this domain.
Acknowledgment
This research was supported by NIH/BIBIB R01EB293388, by NIH/NCI R01CA280134, by the Eric
& Wendy Schmidt Fund for AI Research & Innovation, and by the Kemper Marley Foundation.
References
[1] Stephen J Gardner, Joshua Kim, and Indrin J Chetty. Modern radiation therapy planning and
delivery.Hematology/Oncology Clinics, 33(6):947–962, 2019.
[2] Wei Deng, Yunze Yang, Chenbin Liu, Martin Bues, Radhe Mohan, William W Wong, Robert H
Foote, Samir H Patel, and Wei Liu. A critical review of let-based intensity-modulated proton
therapy plan evaluation and optimization for head and neck cancer management.International
Journal of Particle Therapy, 8(1):36–49, 2021.
[3] Xiaodong Zhang, Wei Liu, Yupeng Li, Xiaoqiang Li, Michelle Quan, Radhe Mohan, Aman Anand,
Narayan Sahoo, Michael Gillin, and Xiaorong R Zhu. Parameterization of multiple bragg curves
for scanning proton beams using simultaneous fitting of multiple curves.Physics in Medicine &
Biology, 56(24):7725, 2011.
[4] Steven E Schild, William G Rule, Jonathan B Ashman, Sujay A Vora, Sameer Keole, Aman
Anand, Wei Liu, and Martin Bues. Proton beam therapy for locally advanced lung cancer: A
review.World journal of clinical oncology, 5(4):568, 2014.
12

[5] Heng Li, Yupeng Li, Xiaodong Zhang, Xiaoqiang Li, Wei Liu, Michael T Gillin, and X Ronald
Zhu. Dynamically accumulated dose and 4d accumulated dose for moving tumors.Medical physics,
39(12):7359–7367, 2012.
[6] Jie Shan, Yunze Yang, Steven E Schild, Thomas B Daniels, William W Wong, Mirek Fatyga,
Martin Bues, Terence T Sio, and Wei Liu. Intensity-modulated proton therapy (impt) interplay
effect evaluation of asymmetric breathing with simultaneous uncertainty considerations in patients
with non-small cell lung cancer.Medical physics, 47(11):5428–5440, 2020.
[7] Victor Hernandez, Christian Rønn Hansen, Lamberto Widesott, Anna B¨ ack, Richard Canters,
Marco Fusella, Julia G¨ otstedt, Diego Jurado-Bruggeman, Nobutaka Mukumoto, Laura Patricia
Kaplan, et al. What is plan quality in radiotherapy? the importance of evaluating dose metrics,
complexity, and robustness of treatment plans.Radiotherapy and Oncology, 153:26–33, 2020.
[8] Mark W Geurts, Dustin J Jacqmin, Lindsay E Jones, Stephen F Kry, Dimitris N Mihailidis,
Jared D Ohrt, Timothy Ritter, Jennifer B Smilowitz, and Nicholai E Wingreen. Aapm med-
ical physics practice guideline 5. b: Commissioning and qa of treatment planning dose calcu-
lations—megavoltage photon and electron beams.Journal of applied clinical medical physics,
23(9):e13641, 2022.
[9] Alexandra O Leone, Abdallah SR Mohamed, Clifton D Fuller, Christine B Peterson, Adam S
Garden, Anna Lee, Lauren L Mayo, Amy C Moreno, Jay P Reddy, Karen Hoffman, et al. A
visualization and radiation treatment plan quality scoring method for triage in a population-based
context.Advances in Radiation Oncology, 9(8):101533, 2024.
[10] Qingxin Wang, Zhongqiu Wang, Minghua Li, Xinye Ni, Rong Tan, Wenwen Zhang, Maitudi
Wubulaishan, Wei Wang, Zhiyong Yuan, Zhen Zhang, et al. A feasibility study of automat-
ing radiotherapy planning with large language model agents.Physics in Medicine & Biology,
70(7):075007, 2025.
[11] Mohammad Hussein, Ben JM Heijmen, Dirk Verellen, and Andrew Nisbet. Automation in in-
tensity modulated radiotherapy treatment planning—a review of recent innovations.The British
journal of radiology, 91(1092):20180270, 2018.
[12] Wenhua Cao, Mary Gronberg, Adenike Olanrewaju, Thomas Whitaker, Karen Hoffman, Carlos
Cardenas, Adam Garden, Heath Skinner, Beth Beadle, and Laurence Court. Knowledge-based
planning for the radiation therapy treatment plan quality assurance for patients with head and
neck cancer.Journal of applied clinical medical physics, 23(6):e13614, 2022.
[13] Christian Rønn Hansen, Mohammad Hussein, Uffe Bernchou, Ruta Zukauskaite, and David
Thwaites. Plan quality in radiotherapy treatment planning–review of the factors and challenges.
Journal of Medical Imaging and Radiation Oncology, 66(2):267–278, 2022.
[14] Ganeshkumar Patel, Abhijit Mandal, Ravindra Shende, and Avinav Bharati. Radiotherapy plan
evaluation indices: A dosimetrical suitability check.Journal of Cancer Research and Therapeutics,
17(2):455–462, 2021.
[15] Wei Liu, Samir H Patel, Daniel P Harrington, Yanle Hu, Xiaoning Ding, Jiajian Shen, Michele Y
Halyard, Steven E Schild, William W Wong, Gary E Ezzell, et al. Exploratory study of the
association of volumetric modulated arc therapy (vmat) plan robustness with local failure in head
and neck cancer.Journal of applied clinical medical physics, 18(4):76–83, 2017.
[16] Wei Liu, Samir H Patel, Jiajian Jason Shen, Yanle Hu, Daniel P Harrington, Xiaoning Ding,
Michele Y Halyard, Steven E Schild, William W Wong, Gary A Ezzell, et al. Robustness quantifi-
cation methods comparison in volumetric modulated arc therapy to treat head and neck cancer.
Practical radiation oncology, 6(6):e269–e275, 2016.
[17] Yixiu Kang, Jiajian Shen, Wei Liu, Paige A Taylor, Hunter S Mehrens, Xiaoning Ding, Yanle
Hu, Erik Tryggestad, Sameer R Keole, Steven E Schild, et al. Impact of planned dose reporting
methods on gamma pass rates for iroc lung and liver motion phantoms treated with pencil beam
scanning protons.Radiation Oncology, 14(1):108, 2019.
13

[18] Chenbin Liu, Zhengliang Liu, Jason Holmes, Lu Zhang, Lian Zhang, Yuzhen Ding, Peng Shu,
Zihao Wu, Haixing Dai, Yiwei Li, et al. Artificial general intelligence for radiation oncology.
Meta-radiology, 1(3):100045, 2023.
[19] Niklas Wahl, Philipp Hennig, Hans-Peter Wieser, and Mark Bangert. Analytical probabilistic
modeling of dose-volume histograms.Medical physics, 47(10):5260–5273, 2020.
[20] Charles S Mayo, John Yao, Avraham Eisbruch, James M Balter, Dale W Litzenberg, Martha M
Matuszak, Marc L Kessler, Grant Weyburn, Carlos J Anderson, Dawn Owen, et al. Incorporating
big data into treatment plan evaluation: Development of statistical dvh metrics and visualization
dashboards.Advances in radiation oncology, 2(3):503–514, 2017.
[21] Lovisa Engberg, Anders Forsgren, Kjell Eriksson, and Bj¨ orn H˚ ardemark. Explicit optimization
of plan quality measures in intensity-modulated radiation therapy treatment planning.Medical
Physics, 44(6):2045–2053, 2017.
[22] Tianfang Zhang, Rasmus Bokrantz, and Jimmy Olsson. Direct optimization of dose–volume
histogram metrics in radiation therapy treatment planning.Biomedical Physics & Engineering
Express, 6(6):065018, 2020.
[23] Anil Pyakuryal, W Kenji Myint, Mahesh Gopalakrishnan, Sunyoung Jang, Jerilyn A Logemann,
and Bharat B Mittal. A computational tool for the efficient analysis of dose-volume histograms for
radiation therapy treatment plans.Journal of Applied Clinical Medical Physics, 11(1):137–157,
2010.
[24] Munib Mesinovic, Peter Watkinson, and Tingting Zhu. Explainability in the age of large language
models for healthcare.Communications Engineering, 4(1):128, 2025.
[25] Di Hu, Yawen Guo, Yiliang Zhou, Lidia Flores, and Kai Zheng. A systematic review of early
evidence on generative ai for drafting responses to patient messages.npj Health Systems, 2(1):27,
2025.
[26] Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, and Heng Ji.
Mint: Evaluating llms in multi-turn interaction with tools and language feedback.arXiv preprint
arXiv:2309.10691, 2023.
[27] H. Dai, Z. Liu, W. Liao, X. Huang, Y. Cao, Z. Wu, L. Zhao, S. Xu, F. Zeng, W. Liu, N. Liu,
S. Li, D. Zhu, H. Cai, L. Sun, Q. Li, D. Shen, T. Liu, and X. Li. Auggpt: Leveraging chatgpt for
text data augmentation.IEEE Transactions on Big Data, 11(03):907–918, 2025.
[28] Yuexing Hao, Jason Holmes, Jared Hobson, Alexandra Bennett, Elizabeth L. McKone, Daniel K.
Ebner, David M. Routman, Satomi Shiraishi, Samir H. Patel, Nathan Y. Yu, Chris L. Hallemeier,
Brooke E. Ball, Mark Waddle, and Wei Liu. Retrospective comparative analysis of prostate cancer
in-basket messages: Responses from closed-domain large language models versus clinical teams.
Mayo Clinic Proceedings: Digital Health, 3(1):100198, 2025.
[29] Yuexing Hao, Zhiwen Qiu, Jason Holmes, Corinna E. L¨ ockenhoff, Wei Liu, Marzyeh Ghassemi,
and Saleh Kalantari. Large language model integrations in cancer decision-making: a systematic
review and meta-analysis.npj Digital Medicine, 8(1):450, 2025.
[30] J. Holmes, L. Zhang, Y. Ding, H. Feng, Z. Liu, T. Liu, W. W. Wong, S. A. Vora, J. B. Ashman,
and W. Liu. Benchmarking a foundation large language model on its ability to relabel structure
names in accordance with the american association of physicists in medicine task group-263 report.
Pract Radiat Oncol, 2024.
[31] Z. Liu, M. He, Z. Jiang, Z. Wu, H. Dai, L. Zhang, S. Luo, T. Han, X. Li, X. Jiang, D. Zhu, X. Cai,
B. Ge, W. Liu, J. Liu, D. Shen, and T. Liu. Survey on natural language processing in medical
image analysis.Zhong Nan Da Xue Xue Bao Yi Xue Ban, 47(8):981–993, 2022.
[32] Zhengliang Liu, Yiwei Li, Peng Shu, Aoxiao Zhong, Hanqi Jiang, Yi Pan, Longtao Yang, Chao Ju,
Zihao Wu, Chong Ma, Cheng Chen, Sekeun Kim, Haixing Dai, Lin Zhao, Lichao Sun, Dajiang Zhu,
Jun Liu, Wei Liu, Dinggang Shen, Quanzheng Li, Tianming Liu, and Xiang Li. Radiology-gpt:
A large language model for radiology.Meta-Radiology, 3(2):100153, 2025.
14

[33] Z. Liu, L. Zhang, Z. Wu, X. Yu, C. Cao, H. Dai, N. Liu, J. Liu, W. Liu, Q. Li, D. Shen, X. Li,
D. Zhu, and T. Liu. Surviving chatgpt in healthcare.Front Radiol, 3:1224682, 2023.
[34] Zhengliang Liu, Aoxiao Zhong, Yiwei Li, Longtao Yang, Chao Ju, Zihao Wu, Chong Ma, Peng
Shu, Cheng Chen, Sekeun Kim, Haixing Dai, Lin Zhao, Dajiang Zhu, Jun Liu, Wei Liu, Dinggang
Shen, Quanzheng Li, Tianming Liu, and Xiang Li. Tailoring large language models to radiology:
A preliminary approach to llm adaptation for a highly specialized domain. In Xiaohuan Cao,
Xuanang Xu, Islem Rekik, Zhiming Cui, and Xi Ouyang, editors,Machine Learning in Medical
Imaging, pages 464–473. Springer Nature Switzerland, 2024.
[35] Saed Rezayi, Haixing Dai, Zhengliang Liu, Zihao Wu, Akarsh Hebbar, Andrew H. Burns, Lin Zhao,
Dajiang Zhu, Quanzheng Li, Wei Liu, Sheng Li, Tianming Liu, and Xiang Li. Clinicalradiobert:
Knowledge-infused few shot learning for clinical notes named entity recognition. In Chunfeng
Lian, Xiaohuan Cao, Islem Rekik, Xuanang Xu, and Zhiming Cui, editors,Machine Learning in
Medical Imaging, pages 269–278. Springer Nature Switzerland, 2022.
[36] Peilong Wang, Jason Holmes, Zhengliang Liu, Dequan Chen, Tianming Liu, Jiajian Shen, and
Wei Liu. A recent evaluation on the performance of llms on radiation oncology physics using
questions of randomly shuffled options.Frontiers in Oncology, Volume 15 - 2025, 2025.
[37] Mikhail Budnikov, Anna Bykova, and Ivan P Yamshchikov. Generalization potential of large
language models.Neural Computing and Applications, 37(4):1973–1997, 2025.
[38] Alex J Goodell, Simon N Chu, Dara Rouholiman, and Larry F Chu. Large language model agents
can use tools to perform clinical calculations.npj Digital Medicine, 8(1):163, 2025.
[39] Humza Nusrat, Bing Luo, Ryan Hall, Joshua Kim, Hassan Bagher-Ebadian, Anthony Doemer,
Benjamin Movsas, and Kundan Thind. Autonomous radiotherapy treatment planning using dola:
A privacy-preserving, llm-based optimization agent.arXiv preprint arXiv:2503.17553, 2025.
[40] Lameck Mbangula Amugongo, Pietro Mascheroni, Steven Brooks, Stefan Doering, and Jan Seidel.
Retrieval augmented generation for large language models in healthcare: A systematic review.
PLOS Digital Health, 4(6):e0000877, 2025.
[41] Akihiko Wada, Yuya Tanaka, Mitsuo Nishizawa, Akira Yamamoto, Toshiaki Akashi, Akifumi
Hagiwara, Yayoi Hayakawa, Junko Kikuta, Keigo Shimoji, Katsuhiro Sano, et al. Retrieval-
augmented generation elevates local llm quality in radiology contrast media consultation.npj
Digital Medicine, 8(1):395, 2025.
[42] Sheng Liu, Oscar Pastor-Serrano, Yizheng Chen, Matthew Gopaulchan, Weixin Liang, Mark
Buyyounouski, Erqi Pollom, Quynh-Thu Le, Michael Francis Gensheimer, Peng Dong, et al.
Automated radiotherapy treatment planning guided by gpt-4vision.Physics in Medicine and
Biology, 2024.
[43] Kehan Xu, Kun Zhang, Jingyuan Li, Wei Huang, and Yuanzhuo Wang. Crp-rag: A retrieval-
augmented generation framework for supporting complex logical reasoning and knowledge plan-
ning.Electronics, 14(1):47, 2024.
[44] Zhengliang Liu, Peilong Wang, Yiwei Li, Jason Holmes, Peng Shu, Lian Zhang, Chenbin Liu,
Ninghao Liu, Dajiang Zhu, Xiang Li, et al. Radonc-gpt: A large language model for radiation
oncology.arXiv preprint arXiv:2309.10160, 2023.
[45] Maria Frasca, Davide La Torre, Gabriella Pravettoni, and Ilaria Cutica. Explainable and inter-
pretable artificial intelligence in medicine: a systematic bibliometric review.Discover Artificial
Intelligence, 4(1):15, 2024.
[46] Elisabeth Hildt. What is the role of explainability in medical artificial intelligence? a case-based
approach.Bioengineering, 12(4):375, 2025.
[47] Se Young Kim, Dae Ho Kim, Min Ji Kim, Hyo Jin Ko, and Ok Ran Jeong. Xai-based clinical
decision support systems: a systematic review.Applied Sciences, 14(15):6638, 2024.
15

[48] Boris Campillo-Gimenez, Wassim Jouini, Sahar Bayat, and Marc Cuggia. Improving case-based
reasoning systems by combining k-nearest neighbour algorithm with logistic regression in the
prediction of patients’ registration on the renal transplant waiting list.PLoS One, 8(9):e71991,
2013.
[49] Gw´ enol´ e Abgrall, Andre L Holder, Zaineb Chelly Dagdia, Karine Zeitouni, and Xavier Monnet.
Should ai models be explainable to clinicians?Critical Care, 28(1):301, 2024.
[50] Yohei Okada, Yilin Ning, and Marcus Eng Hock Ong. Explainable artificial intelligence in emer-
gency medicine: an overview.Clinical and experimental emergency medicine, 10(4):354, 2023.
16