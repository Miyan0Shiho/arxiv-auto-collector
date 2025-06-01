# Data-efficient Meta-models for Evaluation of Context-based Questions and Answers in LLMs

**Authors**: Julia Belikova, Konstantin Polev, Rauf Parchiev, Dmitry Simakov

**Published**: 2025-05-29 09:50:56

**PDF URL**: [http://arxiv.org/pdf/2505.23299v1](http://arxiv.org/pdf/2505.23299v1)

## Abstract
Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems
are increasingly deployed in industry applications, yet their reliability
remains hampered by challenges in detecting hallucinations. While supervised
state-of-the-art (SOTA) methods that leverage LLM hidden states -- such as
activation tracing and representation analysis -- show promise, their
dependence on extensively annotated datasets limits scalability in real-world
applications. This paper addresses the critical bottleneck of data annotation
by investigating the feasibility of reducing training data requirements for two
SOTA hallucination detection frameworks: Lookback Lens, which analyzes
attention head dynamics, and probing-based approaches, which decode internal
model representations. We propose a methodology combining efficient
classification algorithms with dimensionality reduction techniques to minimize
sample size demands while maintaining competitive performance. Evaluations on
standardized question-answering RAG benchmarks show that our approach achieves
performance comparable to strong proprietary LLM-based baselines with only 250
training samples. These results highlight the potential of lightweight,
data-efficient paradigms for industrial deployment, particularly in
annotation-constrained scenarios.

## Full Text


<!-- PDF content starts -->

arXiv:2505.23299v1  [cs.CL]  29 May 2025Data-efficient Meta-models for Evaluation of Context-based
Questions and Answers in LLMs
Julia Belikova
Sber AI Lab
Moscow, Russia
Moscow Institute of Physics and Technology
Dolgoprudny, Russia
ju.belikova@gmail.comKonstantin Polev∗
Sber AI Lab
Moscow, Russia
endless.dipole@gmail.com
Rauf Parchiev
Sber AI Lab
Moscow, Russia
rauf.parchiev@gmail.comDmitry Simakov
Sber AI Lab
Moscow, Russia
dmitryevsimakov@gmail.com
Abstract
Large Language Models (LLMs) and Retrieval-Augmented Gener-
ation (RAG) systems are increasingly deployed in industry appli-
cations, yet their reliability remains hampered by challenges in
detecting hallucinations. While supervised state-of-the-art (SOTA)
methods that leverage LLM hidden states—such as activation trac-
ing and representation analysis—show promise, their dependence
on extensively annotated datasets limits scalability in real-world
applications. This paper addresses the critical bottleneck of data
annotation by investigating the feasibility of reducing training
data requirements for two SOTA hallucination detection frame-
works: Lookback Lens, which analyzes attention head dynamics,
and probing-based approaches, which decode internal model rep-
resentations. We propose a methodology combining efficient clas-
sification algorithms with dimensionality reduction techniques to
minimize sample size demands while maintaining competitive per-
formance. Evaluations on standardized question-answering RAG
benchmarks show that our approach achieves performance compa-
rable to strong proprietary LLM-based baselines with only 250 train-
ing samples. These results highlight the potential of lightweight,
data-efficient paradigms for industrial deployment, particularly in
annotation-constrained scenarios.
CCS Concepts
•Computing methodologies →Natural language generation .
Keywords
retrieval-augmented generation; question-answering; hallucination
detection; data efficiency; model probing
∗Corresponding author
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
SIGIR ’25, Padua, Italy
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-1592-1/2025/07
https://doi.org/10.1145/3726302.3731969ACM Reference Format:
Julia Belikova, Konstantin Polev, Rauf Parchiev, and Dmitry Simakov. 2025.
Data-efficient Meta-models for Evaluation of Context-based Questions
and Answers in LLMs. In Proceedings of the 48th International ACM SI-
GIR Conference on Research and Development in Information Retrieval (SIGIR
’25), July 13–18, 2025, Padua, Italy. ACM, New York, NY, USA, 5 pages.
https://doi.org/10.1145/3726302.3731969
1 Introduction
Large Language Models (LLMs) and Retrieval-Augmented Gen-
eration (RAG) systems have rapidly become key components in
diverse industry applications, including customer support automa-
tion and enterprise knowledge management. Despite their growing
adoption, the reliability of these systems remains compromised by
hallucinations: outputs that appear plausible but are factually incor-
rect or unsupported by the provided context. Such hallucinations
pose significant risks in high-stakes domains such as healthcare,
financial services, and legal advisory, undermining users’ general
trust.
Current state-of-the-art methods for hallucination detection ex-
ploit LLM hidden state representations through sophisticated tech-
niques such as attention-based activation tracing [ 5] or internal
representation probing [28]. However, these powerful approaches
typically depend on large-scale annotated datasets or intensive
computations on proprietary LLMs, limiting practical deployment.
In practice, industrial deployments face three critical constraints.
Limited Annotated Data: Specialized domain data frequently
requires costly and time-consuming manual annotation, restricting
the availability of labeled examples. Computational Efficiency:
Proprietary LLMs (e.g., GPT-4) often introduce prohibitive latency
and sample processing costs, impeding real-time deployments. Pri-
vacy and Data Sovereignty: Sensitive enterprise data often cannot
be sent to external APIs, making the reliance on locally executable
open-source models important.
Existing approaches are still struggling to collectively address
these constraints, creating a considerable gap between academic
benchmarks and feasible industrial solutions. To bridge this gap, our
research investigates practical strategies for significantly improving
data efficiency and achievable quality without sacrificing simplicity
or computational efficiency.

SIGIR ’25, July 13–18, 2025, Padua, Italy Julia Belikova, Konstantin Polev, Rauf Parchiev, and Dmitry Simakov
Our key contributions specifically include:
(1)We present a framework that adapts the most effective prob-
ing techniques via multi-strategy feature extraction with
dimensionality reduction and effective tabular classifiers,
and provide a protocol for evaluating hallucination detection
methods across the full data scarcity spectrum (50-1000 sam-
ples) on question-answering RAG datasets. Through com-
prehensive experiments, we quantitatively establish the crit-
ical points for preserving quality and provide remarkable
insights into the applicability of the methods.
(2)We demonstrate the effectiveness of TabPFNv2 [ 13] – a tab-
ular foundation model leveraging in-context learning – to
achieve state-of-the-art or competitive hallucination detec-
tion performance under limited data scenarios (see Table 1).
To our knowledge, we are among the first to rigorously ex-
plore the adaptability of advanced tabular foundation models
(TabPFNv2) within hallucination detection tasks, highlight-
ing valuable synergies between foundational tabular meth-
ods and NLP classification applications.
(3)We demonstrate that employing relatively small LLMs as ex-
tractor models for hallucination detection in some scenarios
is comparable or even outperforms specialized evaluation
frameworks based on proprietary models.
Classifier EManual ExpertQA RAGTruth Average Rank
tabpfn 0.7161 0.8204 0.8139 0.7834 1
logreg 0.6896 0.8218 0.8087 0.7734 2
catboost 0.6832 0.7932 0.8176 0.7646 3
att.-pool. probe 0.6776 0.7611 0.8002 0.7463 4
Table 1: Average ROC-AUC scores, corresponding classifier
rankings, based on average on all datasets.
The paper is structured as follows: Section 2 provides an overview
of related work, Section 3 describes our proposed unified detection
framework, Section 4 presents experimental design and compre-
hensive experimental analysis, and Section 5 concludes the paper
with a summary of contributions and future research directions.
2 Related Work
LLM hallucinations and their detection. LLMs exhibit several
types of hallucinations [ 15,26].Factual hallucinations [33] involve
generating information that conflicts with established facts. Seman-
tic distortion [30] refers to errors where the meaning of a word
or phrase is misrepresented. Contextual hallucinations [33] occur
when the model’s response is not consistent with a correctly pro-
vided context. In this work, the focus is specifically on contex-
tual hallucinations in one of the most common industrial tasks –
question-answering (QA), where the question is augmented with
grounding context. Detecting such hallucinations is a crucial step
toward hallucination reduction in industrial systems, especially
for applications requiring exact factual answers, such as medical
[10, 20] or financial [4, 27, 34] RAG systems.
This focus on contextual hallucination detection has led to the
development of strong context-based methods such as Lookback
Lens, introduced in [ 5], which measures the lookback ratio of atten-
tion weights between the provided context and newly generated
tokens, applying a linear classifier to these features. Other work[23,28] shows the high effectiveness of different probing meth-
ods for domain-specific learning, such as [28], which trains linear ,
attention-pooling , and ensemble probes over the outputs of a trans-
former block [ 31]. A parallel line of research explores uncertainty
estimation for hallucination detection [ 6,8,16,19]. These methods
are rapidly evolving but still may struggle with calibration issues,
require multiple forward passes, or may not capture the nuanced
relationship between model confidence and factual accuracy [ 14].
Therefore, in this work, we focus on lightweight probing techniques
that have proven to be effective in industrial domains [2].
The recent advancements in LLMs [ 1,17] have also enabled re-
searchers to use LLMs as judges to evaluate other models’ answers
[35]. Notably, RAGAS [ 7], a comprehensive evaluation system that
uses an LLM evaluator, is specifically designed for RAG systems.
While these and other frameworks like [ 3,25] and TruLens1have
introduced powerful methods for the contextual hallucination detec-
tion task, they typically require access to multiple model responses,
large amounts of data, or access to proprietary models. In contrast,
our methodology is designed for data- and compute-constrained
settings, making it more suitable for industrial deployment.
In-context learning for tabular data. Recent advances in deep
learning have extended to tabular tasks, traditionally solved using
linear or gradient boosting algorithms [ 24]. In-context learning
approaches are particularly effective in low-data scenarios, such as
when limited annotated data is available for hallucination detection.
TabPFN [ 12,13], a tabular foundation model designed for small
datasets, has shown promise in this area, outperforming previous
methods while requiring significantly less computation time. This
paper investigates TabPFN’s applicability to hallucination detection
and compares its performance with other tabular classifiers.
3 Framework
Our framework introduces a unified and data-efficient approach
to contextual hallucination detection in question-answering tasks.
This two-stage pipeline combines feature extraction from LLM
internals with lightweight classification to maximize detection per-
formance while minimizing training data requirements.
Feature extraction . An annotator LLM generates an answer to
a contextual augmented question, while an independent extractor
LLM computes internal activations – specifically hidden states
(outputs of transformer blocks) and attention scores. This separation
is necessary when the hidden states of the generator model are
inaccessible. Further, we employ two complementary approaches
to aggregate and compress the extracted activations:
(1)For each sequence of per-token hidden states, we apply mean
and max pooling across the feature dimension and extract the
last token’s hidden state, which research has shown to be par-
ticularly informative for hallucination detection [ 28]. These
three components are taken from the middle layer hidden
states (based on empirical findings from [ 5]) and compressed
separately using dimensionality reduction techniques (PCA
or UMAP) to a fixed size, a constraint motivated by our goal
of minimal training data requirements. The resulting vec-
tors are then concatenated to form a comprehensive feature
representation.
1https://www.trulens.org

Data-efficient Meta-models for Evaluation of Context-based Questions and Answers in LLMs SIGIR ’25, July 13–18, 2025, Padua, Italy
(2)Building on the Lookback Lens methodology, we compute
the ratio of attention weights between the provided context
and newly generated tokens. Although the original imple-
mentation applies this analysis to specific token spans, we
adapt it by computing the mean lookback ratio across the en-
tire sequence, making it more suitable for our data-efficient
paradigm. Given the high dimensionality of lookback ra-
tios (layers ×attention heads), we propose two alternative
feature extraction strategies: applying dimensionality reduc-
tion similar to the pooling probe approach and selectively
using the middle layers of the model with all their attention
heads while ensuring the feature count remains within the
input constraints of our tabular classifiers (particularly the
TabPFNv2’s 500-feature limit).
Meta-Classification . The extracted feature vectors are fed into
lightweight meta-classifiers specifically chosen for their effective-
ness in low-data settings. We evaluated three primary classification
approaches: logistic regression, CatBoost [24], and TabPFN2.
4 Experiments
4.1 Experimental Setup
Datasets . Three datasets that reflect real-world industrial con-
straints are used in this study: RAGTruth [ 22] (QA subset), and
ExpertQA [ 18] and EManual [ 21] datasets from RAGBench [ 9].
RAGTruth is a human-annotated dataset with labels curated by
domain experts, whereas RAGBench uses GPT-4 as the annotator
to generate high-quality, scalable, albeit synthetic, hallucination
labels. Examples of context-based question-answer pairs from these
datasets are shown in Table 2.
RAGTruth :hallucinated answer
Question : Risks benefits roth ira vs 401k
Context : Passage [1]: A Roth 401(k) account works much like a Roth
IRA, but there is no income limit to participate, and you are required to
take the minimum distributions ...
Answer : Based on the given passages, the benefits of a Roth IRA include
the lack of income limits to participate, the ability to save more after
maxing out the Roth IRA contribution, and the higher contribution
limits compared to a Roth IRA ...
ExpertQA :correct answer
Question : What is the best way to store my medication to ensure its
potency and safety?
Context : Document [1]: Maintaining the right temperature in a phar-
macy refrigerator is crucial to ensure the safe storage of medication ...
Answer : The best way to store medication to ensure its potency and
safety is to follow the storage instructions provided ...
Table 2: Examples of context-based question-answer pairs
from RAGTruth and ExpertQA datasets.
Training and Evaluation . The performance of methods is eval-
uated under the available training data size constraint. For RAG-
Bench, training subsets of sizes 50, 100, 250, 500, 750, and 1000 are
sampled, while for RAGTruth, we explore sizes 50, 100, 250, and 500.
In each case, we hold out 20% of the training samples for validation
and use the default test splits. Each experiment is repeated withdifferent random seeds three times to ensure robust performance
estimates.
We use ROC-AUC as our primary evaluation metric, as it is in-
dependent of threshold and provides a robust measure of detection
performance across different operating points. In addition, we re-
port mean reciprocal rank (MRR) to evaluate how well our models
prioritize the detection of hallucinations when ranking multiple
responses (see Table 3).
Classifiergemma-2-
9b-itLlama-3.1-
8B-InstructQwen2.5-
7B-Instruct
att.-pool. probe 0.2778 0.3611 0.3333
catboost 0.4444 0.3889 0.5556
logreg 0.7778 0.5 0.4167
tabpfn 0.5833 0.8333 0.7778
Table 3: Mean reciprocal rank across datasets for the detec-
tors. Higher values indicate superior performance.
Baselines . For comparison, the standard attention-pooling probe
is evaluated against the proposed modifications. In addition, we
evaluate the GPT-4o zero-shot judge and specialized prompting of
the RAGAS framework with the GPT-4o model for the faithfulness
assessment. As explained above, these approaches do not satisfy the
constraints of the industry-related setting we consider but serve as
practical upper bounds in our experiments.
Main methods . For the methods in the proposed framework,
we use three smaller open-source LLMs as extractors of internal
activations: Gemma-2-9B-It [ 29], Llama-3.1-8B [ 11], and Qwen2.5-
7B-Instruct [ 32]. When the dimensionality reduction step is applied,
each feature type – namely pooled hidden states and lookback ra-
tios – is reduced to 30 components to accommodate training size
limitations. This ensures the total feature count remains below
the 500 limit required for TabPFNv2 compatibility. Alternatively,
when dimensionality reduction is omitted for lookback ratios, spe-
cific layer ranges (Qwen: 5–21, Llama: 8–22, Gemma: 5–35) are
selected to approximate this 500-feature ceiling, while accounting
for variations in attention head counts.
4.2 Results and Analysis
The analysis explores findings across five directions: (1) overall
effectiveness concerning strong external baselines, (2) impact of
feature design and dimensionality reduction, (3) choice of meta-
classifier, (4) choice of extractor LLM, and (5) data-efficiency. All
numbers are averaged over three random seeds and are shown in
Figure 1 for the main set of methods. Table 4 gives a brief cross-
extractor snapshot of detector quality before we examine the results
in detail.
Classifiergemma-2-
9b-itLlama-3.1-
8B-InstructQwen2.5-
7B-Instruct
att.-pool. probe 0.6379 0.7226 0.7199
catboost 0.719 0.7328 0.7466
logreg 0.7334 0.7378 0.7505
tabpfn 0.7242 0.7587 0.7623
Table 4: Mean ROC-AUC across datasets for the detectors.

SIGIR ’25, July 13–18, 2025, Padua, Italy Julia Belikova, Konstantin Polev, Rauf Parchiev, and Dmitry Simakov
Figure 1: Test ROC -AUC versus training -set size for the pro-
posed evaluators (solid lines) across the three benchmarks
(rows) and three response generators (columns). Horizontal
dashed lines correspond to the zero -shot GPT -4o judge (yel-
low) and the RAGAS GPT -4o pipeline (cyan). Shaded areas
indicate ±95% confidence intervals over three random seeds.
Method EManual ExpertQA RAGTruth
PCA + lookback + tabpfn 0.6972 0.8165 0.7679
lookback + tabpfn 0.6659 0 .8064 0.8037
lookback + logreg 0.6608 0 .8122 0 .7828
att.-pool. probe 0.6776 0 .7611 0 .8002
PCA + hidden + tabpfn 0.6266 0 .8115 0 .7944
PCA + lookback + catboost 0.6736 0 .7785 0 .7717
PCA + lookback + logreg 0.6872 0.8051 0 .7292
PCA + hidden + catboost 0.6336 0 .7887 0 .7801
PCA + hidden + logreg 0.6018 0.8143 0.7643
lookback + catboost 0.5898 0 .7792 0.8028
UMAP + hidden + tabpfn 0.6714 0 .7758 0 .7131
UMAP + lookback + logreg 0.5960 0 .7577 0 .7701
UMAP + hidden + catboost 0.6479 0 .7672 0 .6987
UMAP + hidden + logreg 0.6439 0 .7723 0 .6899
UMAP + lookback + tabpfn 0.5941 0 .7529 0 .7462
UMAP + lookback + catboost 0.6226 0 .7324 0 .6941
RAGAS GPT-4o 0.8208 0 .8160 0 .8386
GPT-4o 0.6227 0 .7423 0 .7000
Table 5: Mean ROC-AUC scores across extractor models of
various combinations of feature extraction, dimensional-
ity reduction, and classification algorithms, along with the
scores from two types of LLM judges.
Overall effectiveness. Our experiments demonstrate that light-
weight methods leveraging internal states of LLMs can achieve
substantial hallucination detection performance, even with limited
training data. As aggregated in Table 5, the best-performing meth-
ods, especially those combining lookback features with TabPFN,
approach (for RAGTruth) or, in some cases, even surpass (ExpertQA)
the performance of the strong RAGAS GPT-4o baseline. While RA-
GAS GPT-4o generally sets a high benchmark (ROC-AUC 0.81-0.84),the proposed methods reach competitive levels (up to 0.81 ROC-
AUC). They significantly outperform zero-shot GPT-4o, validating
their potential for efficient local hallucination detection.
Impact of feature design and dimensionality reduction.
PCA dimensionality reduction often enhances lookback feature
performance, proving more effective than UMAP or using raw
features in most configurations.
Choice of meta-classifier. TabPFN is the top-performing meta-
classifier on average ROC-AUC and MRR, demonstrating effective-
ness in low-data settings (Tables 1, 3, 4). Logistic Regression is
also highly competitive. Interestingly, both, on average, outperform
CatBoost and the baseline attention-pooling probe approach.
Choice of extractor LLM. We observe that the choice of ex-
tractor LLM does influence the results; Llama-3.1-8B and Qwen2.5-
7B, on average, yield superior detector performance compared to
Gemma-2-9B across classifiers (Figure 1).
Data-efficiency. The methods exhibit strong data-efficiency
(Figure 1, Table 6). Performance improves sharply with initial data
(50-250 samples) before plateauing. High performance relative to
baselines is achievable with only 250 examples, confirming suitabil-
ity for annotation-constrained industrial settings.
Train size EManual ExpertQA RAGTruth
50 0.5246 0.6939 0.6604
100 0.5569 0.7294 0.6565
250 0.5903 0.7648 0.7368
500 0.6051 0.7809 0.7655
750 0.6056 0.7959 -
1000 0.6427 0.7985 -
Table 6: Effect of dataset size on quality (ROC-AUC).
In general , we conducted a thorough evaluation involving mul-
tiple combinations of dimensionality reduction techniques, feature
extraction methods, and meta-classifiers. Our experiments indicate
that methods based on TabPFNv2 exhibit strong performance in
data-scarce settings and, on average, outperform other classifiers
we evaluated.
5 Conclusion
The reliability of LLM-driven RAG systems in industry settings
hinges on scalable, cost-effective methods to detect hallucinations.
This work demonstrates that lightweight, annotation-efficient ap-
proaches can achieve competitive performance while overcoming
three critical industrial constraints: limited labeled data, prohibitive
computational costs, and data privacy risks. Proposed data-efficient
methods also highlight novel use cases for TabPFNv2 in NLP do-
mains, reveal limitations of attention-based probing techniques, and
illustrate effective, lightweight heuristic strategies for hallucina-
tion detection. Future research should further explore architectural
modifications of tabular foundation models (e.g., TabPFNv2) specif-
ically optimized for NLP representation learning and contextual
hallucination detection tasks.
Acknowledgments
We thank Kseniia Kuvshinova and Aziz Temirkhanov for valuable
discussions, feedback and their high contribution to this work.

Data-efficient Meta-models for Evaluation of Context-based Questions and Answers in LLMs SIGIR ’25, July 13–18, 2025, Padua, Italy
Presenter Bio
Julia Belikova is an M.Sc. candidate in computer science at the
Moscow Institute of Physics and Technology. She is also an NLP
Researcher at Sber AI Laboratory. Her work focuses on addressing
critical challenges in generative AI, particularly LLM hallucination
detection and uncertainty quantification.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al .2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774
(2023).
[2]Ingeol Baek, Hwan Chang, ByeongJeong Kim, Jimin Lee, and Hwanhee Lee. 2025.
Probing-RAG: Self-Probing to Guide Language Models in Selective Document
Retrieval. In Findings of the Association for Computational Linguistics: NAACL 2025 ,
Luis Chiruzzo, Alan Ritter, and Lu Wang (Eds.). Association for Computational
Linguistics, Albuquerque, New Mexico, 3287–3304. https://aclanthology.org/
2025.findings-naacl.181/
[3]Masha Belyi, Robert Friel, Shuai Shao, and Atindriyo Sanyal. 2025. Luna: A
Lightweight Evaluation Model to Catch Language Model Hallucinations with
High Accuracy and Low Cost. In Proceedings of the 31st International Conference
on Computational Linguistics: Industry Track . 398–409.
[4]Yuemin Chen, Feifan Wu, Jingwei Wang, Hao Qian, Ziqi Liu, Zhiqiang Zhang, Jun
Zhou, and Meng Wang. 2024. Knowledge-augmented Financial Market Analysis
and Report Generation. In Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing: Industry Track . 1207–1217.
[5]Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ranjay Krishna, Yoon Kim, and
James Glass. 2024. Lookback Lens: Detecting and Mitigating Contextual Halluci-
nations in Large Language Models Using Only Attention Maps. In Proceedings
of the 2024 Conference on Empirical Methods in Natural Language Processing .
1419–1436.
[6]Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu,
Bhavya Kailkhura, and Kaidi Xu. 2024. Shifting Attention to Relevance: Towards
the Predictive Uncertainty Quantification of Free-Form Large Language Models.
InProceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , Lun-Wei Ku, Andre Martins, and Vivek
Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand,
5050–5063. doi:10.18653/v1/2024.acl-long.276
[7]Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. Ragas:
Automated evaluation of retrieval augmented generation. In Proceedings of the
18th Conference of the European Chapter of the Association for Computational
Linguistics: System Demonstrations . 150–158.
[8]Marina Fomicheva, Shuo Sun, Lisa Yankovskaya, Frédéric Blain, Francisco
Guzmán, Mark Fishel, Nikolaos Aletras, Vishrav Chaudhary, and Lucia Spe-
cia. 2020. Unsupervised Quality Estimation for Neural Machine Translation.
Transactions of the Association for Computational Linguistics 8 (2020), 539–555.
doi:10.1162/tacl_a_00330
[9]Robert Friel, Masha Belyi, and Atindriyo Sanyal. 2024. Ragbench: Explain-
able benchmark for retrieval-augmented generation systems. arXiv preprint
arXiv:2407.11005 (2024).
[10] Giacomo Frisoni, Miki Mizutani, Gianluca Moro, and Lorenzo Valgimigli. 2022.
BioReader: a Retrieval-Enhanced Text-to-Text Transformer for Biomedical Lit-
erature. In Proceedings of the 2022 conference on empirical methods in natural
language processing . 5770–5793.
[11] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek
Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, et al .2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783
(2024).
[12] Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. 2022.
Tabpfn: A transformer that solves small tabular classification problems in a
second. arXiv preprint arXiv:2207.01848 (2022).
[13] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max
Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. 2025. Accurate
predictions on small data with a tabular foundation model. Nature 637, 8045
(2025), 319–326.
[14] Yuheng Huang, Jiayang Song, Zhijie Wang, Shengming Zhao, Huaming Chen,
Felix Juefei-Xu, and Lei Ma. 2025. Look Before You Leap: An Exploratory Study of
Uncertainty Analysis for Large Language Models. IEEE Transactions on Software
Engineering 51, 2 (2025), 413–429. doi:10.1109/TSE.2024.3519464
[15] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in
natural language generation. Comput. Surveys 55, 12 (2023), 1–38.
[16] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic Uncertainty:
Linguistic Invariances for Uncertainty Estimation in Natural Language Genera-
tion. In The Eleventh International Conference on Learning Representations, ICLR2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net. https://openreview.net/
pdf?id=VD-AYtP0dve
[17] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Cheng-
gang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al .2024. Deepseek-v3
technical report. arXiv preprint arXiv:2412.19437 (2024).
[18] Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth Sieber, Mark Yatskar, and
Dan Roth. 2023. Expertqa: Expert-curated questions and attributed answers.
arXiv preprint arXiv:2309.07852 (2023).
[19] Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. 2023. SelfCheckGPT: Zero-
Resource Black-Box Hallucination Detection for Generative Large Language
Models. arXiv:2303.08896 [cs.CL] https://arxiv.org/abs/2303.08896
[20] Aakanksha Naik, Sravanthi Parasa, Sergey Feldman, Lucy Lu Wang, and Tom
Hope. 2022. Literature-Augmented Clinical Outcome Prediction. In Findings of
the Association for Computational Linguistics: NAACL 2022 . 438–453.
[21] Abhilash Nandy, Soumya Sharma, Shubham Maddhashiya, Kapil Sachdeva,
Pawan Goyal, and Niloy Ganguly. 2021. Question answering over electronic de-
vices: A new benchmark dataset and a multi-task learning based QA framework.
arXiv preprint arXiv:2109.05897 (2021).
[22] Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy Zhong,
Juntong Song, and Tong Zhang. 2023. Ragtruth: A hallucination corpus for
developing trustworthy retrieval-augmented language models. arXiv preprint
arXiv:2401.00396 (2023).
[23] Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas
Kotek, and Yonatan Belinkov. 2024. LLMs Know More Than They Show: On
the Intrinsic Representation of LLM Hallucinations. arXiv:2410.02707 [cs.CL]
https://arxiv.org/abs/2410.02707
[24] Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Doro-
gush, and Andrey Gulin. 2018. CatBoost: unbiased boosting with categorical
features. Advances in neural information processing systems 31 (2018).
[25] Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. 2024.
ARES: An Automated Evaluation Framework for Retrieval-Augmented Genera-
tion Systems. In Proceedings of the 2024 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies
(Volume 1: Long Papers) . 338–354.
[26] Pranab Sahoo, Prabhash Meharia, Akash Ghosh, Sriparna Saha, Vinija Jain, and
Aman Chadha. 2024. A Comprehensive Survey of Hallucination in Large Lan-
guage, Image, Video and Audio Foundation Models. In Findings of the Association
for Computational Linguistics: EMNLP 2024 . 11709–11724.
[27] Bhaskarjit Sarmah, Dhagash Mehta, Benika Hall, Rohan Rao, Sunil Patel, and
Stefano Pasquali. 2024. Hybridrag: Integrating knowledge graphs and vector
retrieval augmented generation for efficient information extraction. In Proceedings
of the 5th ACM International Conference on AI in Finance . 608–616.
[28] CH-Wang Sky, Benjamin Van Durme, Jason Eisner, and Chris Kedzie. 2024. Do
Androids Know They’re Only Dreaming of Electric Sheep?. In Findings of the
Association for Computational Linguistics ACL 2024 . 4401–4420.
[29] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy
Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari,
Alexandre Ramé, et al .2024. Gemma 2: Improving open language models at a
practical size. arXiv preprint arXiv:2408.00118 (2024).
[30] Gabriel Tjio, Ping Liu, Joey Tianyi Zhou, and Rick Siow Mong Goh. 2022. Adver-
sarial semantic hallucination for domain generalized semantic segmentation. In
Proceedings of the IEEE/CVF winter conference on applications of computer vision .
318–327.
[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. Advances in neural information processing systems 30 (2017).
[32] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang,
Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang
Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue,
Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang
Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu,
Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. Qwen2.5 Technical Report. arXiv
preprint arXiv:2412.15115 (2024).
[33] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting
Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al .2023. Siren’s song in the
AI ocean: a survey on hallucination in large language models. arXiv preprint
arXiv:2309.01219 (2023).
[34] Yiyun Zhao, Prateek Singh, Hanoz Bhathena, Bernardo Ramos, Aviral Joshi,
Swaroop Gadiyaram, and Saket Sharma. 2024. Optimizing LLM based retrieval
augmented generation pipelines in the financial domain. In Proceedings of the 2024
Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 6: Industry Track) . 279–294.
[35] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al .2023. Judging
llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information
Processing Systems 36 (2023), 46595–46623.