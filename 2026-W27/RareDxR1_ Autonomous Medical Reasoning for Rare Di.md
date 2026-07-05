# RareDxR1: Autonomous Medical Reasoning for Rare Disease Diagnosis Beyond Human Annotation

**Authors**: Deyang Jiang, Haoran Wu, Ziyi Wang, Yiming Rong, Yunlong Zhao, Ye Jin, Bo Xu

**Published**: 2026-06-30 20:25:53

**PDF URL**: [https://arxiv.org/pdf/2607.00147v1](https://arxiv.org/pdf/2607.00147v1)

## Abstract
Rare disease differential diagnosis is a critical yet arduous clinical task, requiring physicians to identify precise phenotypes from complex, unstructured patient symptoms and execute intricate reasoning within a vast search space. However, existing AI approaches typically rely on pipeline-based phenotype extraction or retrieval-augmented generation, which suffer from critical information loss due to predefined ontologies, retrieval bottlenecks, and a lack of diagnostic logic. To address these challenges, we introduce RareDxR1, an end-to-end reasoning-centric large language model designed for open-domain rare disease diagnosis directly from unstructured clinical notes. We design a progressive end-to-end training framework by synergizing knowledge internalization with autonomous evolutionary learning, thereby bypassing reliance on structured phenotypes and closed-set decision-making. To overcome the limitations of RAG and phenotype restriction, we enabled the deep internalization of fragmented rare-disease knowledge directly into the model's parameters. Moreover, to bridge the gap between model generation and expert reasoning, we propose Reflection-Enhanced Reasoning Sampling (RERS), a strategy that synthesizes expert-level diagnostic trajectories by learning from failures without human annotation. Additionally, we propose a dual-level curriculum reinforcement learning approach for gradually mastering rare disease diagnosis. Experimental results demonstrate that RareDxR1 achieves state-of-the-art accuracy across different benchmarks, marking a significant breakthrough in open-domain rare disease diagnosis. Our code and dataset will be publicly available.

## Full Text


<!-- PDF content starts -->

RareDxR1: Autonomous Medical Reasoning for
Rare Disease Diagnosis Beyond Human Annotation
Deyang Jiang1,2,∗, Haoran Wu1,∗, Ziyi Wang1, Yiming Rong1, Yunlong Zhao1, Ye Jin3, and Bo Xu1,2
1The Key Laboratory of Cognition and Decision Intelligence for Complex Systems,
Institute of Automation, Chinese Academy of Sciences, Beijing, China
2School of Future Technology, University of Chinese Academy of Sciences, Beijing, China
3Peking Union Medical College Hospital, Beijing, China
Abstract—Rare disease differential diagnosis is a critical yet
arduous clinical task, requiring physicians to identify precise
phenotypes from complex, unstructured patient symptoms and
execute intricate reasoning within a vast search space. However,
existing AI approaches typically rely on pipeline-based phenotype
extraction or retrieval-augmented generation, which suffer from
critical information loss due to predefined ontologies, retrieval
bottlenecks, and a lack of diagnostic logic. To address these
challenges, we introduce RareDxR1, an end-to-end reasoning-
centric large language model designed for open-domain rare
disease diagnosis directly from unstructured clinical notes. We
design a progressive end-to-end training framework by syner-
gizing knowledge internalization with autonomous evolutionary
learning, thereby bypassing reliance on structured phenotypes
and closed-set decision-making. To overcome the limitations of
RAG and phenotype restriction, we enabled the deep inter-
nalization of fragmented rare disease knowledge directly into
the model’s parameters. Moreover, to bridge the gap between
model generation and expert reasoning, we propose Reflection-
Enhanced Reasoning Sampling (RERS), a strategy that syn-
thesizes expert-level diagnostic trajectories by learning from
failures without human annotation. Additionally, we propose
a dual-level curriculum reinforcement learning approach for
gradually mastering rare disease diagnosis. Experimental results
demonstrate that RareDxR1 achieves a state-of-the-art accuracy
across different benchmarks, marking a significant breakthrough
in open-domain rare disease diagnosis. Our code and dataset will
be publicly available.1
Index Terms—Rare Disease Diagnosis, Large Language Mod-
els, Reinforcement Learning, Chain-of-Thought, Medical Reason-
ing
∗Equal contribution. Corresponding author: Bo Xu (xubo@ia.ac.cn).
The project is supported by the National High Level Hospital Clinical
Research Funding (2025-PUMCH-C-006), and the National Natural Science
Foundation of China (No. 62506358).
Accepted to the IEEE International Conference on Multimedia and Expo
(ICME) 2026. © 2026 IEEE. Personal use of this material is permitted.
Permission from IEEE must be obtained for all other uses, in any current or
future media, including reprinting/republishing this material for advertising or
promotional purposes, creating new collective works, for resale or redistribu-
tion to servers or lists, or reuse of any copyrighted component of this work
in other works.
The project is supported by the National High Level Hospital Clinical
Research Funding (2025-PUMCH-C-006), and the National Natural Science
Foundation of China (No. 62506358).
1https://github.com/jdy18/RareDxR1
Fig. 1. Top-10 Recall across all rare disease categories in RareArena-Test.
I. INTRODUCTION
Rare diseases, affecting less than 1/2000 of the population,
encompass 7,000–10,000 types globally and impact approxi-
mately 300 million people [1], [2].Rare Disease Diagnosis
(RDD)—the process of identifying specific diseases from
undifferentiated clinical manifestations—is notoriously dif-
ficult. In clinical practice, this task requires a dual-process
capability: first, physicians must meticulously parse unstruc-
tured patient narratives to identify key phenotypes; second,
they must perform complex differential diagnosis, weighing
conflicting evidence against multiple conditions. The scarcity
of specialized expertise often leads to a 4.8-year average
diagnostic delay and frequent misdiagnoses [3], [4].
To assist this decision-making, traditional computational
methods have long relied on standardized phenotype on-
tologies [5]–[8] or genotypic data [9]–[11] as core features.
However, such symbolic inputs depend heavily on manual
feature engineering and often overlook critical nuances em-
bedded in unstructured notes. Furthermore, these methods are
typically confined to predefined, closed label spaces, making
it difficult to recognize unseen diseases or discover potentialarXiv:2607.00147v1  [cs.AI]  30 Jun 2026

rare conditions in open-set clinical practice.
The development of Large Language Models (LLMs) has
made open-domain diagnosis directly from raw text possible.
However, they currently face two critical gaps: insufficient
rare disease knowledge and inadequate clinical reasoning. On
the knowledge front, current Medical-LLMs [12], [13] lack
coverage of long-tail diseases, leading to hallucinations and
a failure to utilize structured phenotypic databases [14], [15].
The reasoning gap is even more severe. General reasoning
models [16], [17] excel in formal logic but struggle with
the probabilistic nature of clinical diagnosis, while Medical-
LLMs lack the expert-level data needed to navigate complex
diagnostic paths. Many researchers employ Rejection Sam-
pling (RS) [18] to synthesize Chain-of-Thought data, but it is
inefficient here: general models rarely generate correct initial
solutions for rare diseases, and discarding incorrect samples
wastes valuable opportunities to learn from failures.
To bridge these gaps, we introduce RareDxR1, an end-to-
end reasoning-centric large language model designed for open-
domain rare disease diagnosis. Unlike previous methods, it
performs direct diagnostic inference from unstructured clinical
notes, without the rigid phenotype extraction step. Moreover, it
achieves expert-level multi-step reasoning capabilities without
requiring any human annotation. Our approach addresses the
challenges through four key technical innovations:
1) To resolve retrieval bottlenecks, we propose a strategy
for Knowledge Internalization. This allows the model
to encode rare disease facts directly into its parameters,
enabling effective ”knowledge emergence” without the
reliance on external databases typical of phenotype-
based or RAG systems.
2) To mitigate the lack of reasoning capacity in data-sparse
scenarios, we introduce Reflection-Enhanced Reasoning
Sampling. It forces the model to reflect on its own
diagnostic failures and correct them.
3) To overcome the limitations of structured phenotype
dependence and closed-set decision-making, we propose
an integrated end-to-end training pipeline and derive
RareDxR1, which enables direct, open-domain diagnosis
from raw clinical notes, achieving SOTA performance on
multiple benchmarks.
4) To integrate the advantages of other models and reduce
model-specific diagnostic biases, we propose Collabo-
rative Reasoning Refinement (CRR). This is a simple
yet effective inference strategy that further improves
diagnostic accuracy by integrating insights from external
models to refine the initial diagnosis.
II. RELATEDWORKS
A. Rare Disease Diagnosis
Traditional approaches, whether phenotype-based [5]–[8]
or gene-based [9]–[11], heavily rely on manually curated
annotations in knowledge bases. Recently, Large Language
Models (LLMs) have been applied via retrieval-augmented
reasoning [19], multi-agent collaboration [20], and tool invo-
cation [21]. While continued medical pre-training [12], [13]has yielded gains, these general-purpose models still struggle
with the sparse, specialized knowledge and intricate reasoning
required for rare disease diagnosis.
B. Reasoning Language Models
Reinforcement learning (RL) has proven effective for en-
hancing reasoning, supported by algorithm optimizations like
GRPO [22] and DAPO [23]. In the medical domain, recent
works such as Med-R1 [24], Med-RLVR [25], and Baichuan-
M1 [12] have utilized RL for alignment. However, these
efforts predominantly focus on general medical QA, leaving
the significantly more complex challenge of open-domain rare
disease differential diagnosis largely unexplored.
III. METHODS
To address the challenges of sparse knowledge and complex
reasoning, we developRareDxR1with a multi-stage training
framework followed by a collaborative inference strategy.
In this section, we first present our data engine designed
for the Supervised Fine-Tuning stage:RareKnowledgeQA
for knowledge integration andRareDxCOTfor expert-level
reasoning synthesis via RERS. Then, we introduceDual-
Level Curriculum Reinforcement Learning (DCRL), a
framework that enables the model to autonomously refine
its diagnostic logic through structured curricula. Finally, we
describeCollaborative Reasoning Refinement (CRR), an
inference-stage method that mitigates single-model biases by
synthesizing multi-model perspectives and retrieved evidence.
A. Task Formulation
We formulate the diagnostic task asF M:C→(r, ˆD),
where modelMgenerates a reasoning trajectoryrand ranked
candidates ˆD= ( ˆd1, . . . , ˆdK)from a clinical noteC, aiming
to rank the ground-truthd∗optimally high.
To mitigate evaluation bias from severe rare disease syn-
onymity, we compiled a multi-source thesaurus consolidating
Orphanet [26], OMIM [15], and ICD [27]. This enables our
synonym-aware matching function, match( ˆd, d∗), which ver-
ifies exact or synonymous matches. Performance is quantified
viaTop-k Recall.
B. Public Data Resources
Structured Knowledge:We integrate HPO [14], Or-
phanet [26], and OMIM [15] to provide standardized encod-
ings for diseases and phenotypes, along with their associations.
Unstructured Knowledge:Open-access medical literature
retrieved from PubMed [28] to supplement textual knowledge.
Rare Disease Cases:We utilize the RareArena dataset [29],
comprising 22,896 annotated clinical notes. To ensure bal-
anced coverage, we performed stratified sampling based on
disease frequency to partition the data into 1,025 test samples
(RareArena-Test), 970 validation samples, and 20,901 train-
ing samples (RareArena-Train) used for RareDxCOT genera-
tion and RL. Additionally, we collected a subset from MIMIC-
IV-Note [30] as an out-of-distribution (OOD) benchmark.

Fig. 2. The two-stage training framework for constructing the RareDxR1 model.
C. Knowledge Internalization (RareKnowledgeQA)
A key challenge in rare disease diagnosis for LLMs lies in
the fragmented nature of knowledge, split between structured
databases and unstructured texts. While structured data ensures
factual accuracy and unstructured text provides rich clinical
context, our preliminary findings indicated that simply expos-
ing LLMs to raw structured data is insufficient for effective
knowledge acquisition.
To address this, we created theRareKnowledgeQAdataset
to unify both knowledge types into an LLM-friendly format.
We extracted core information—disease definitions, pheno-
types, and clinical features—from both sources, and converted
it into question-answer pairs using a hybrid approach: rule-
based templates for factual precision and LLM-based genera-
tion for contextual diversity.
D. Expert-level Reasoning Synthesis (RareDxCOT)
To transition from general reasoning to medical expertise,
we constructedRareDxCOT, a dataset of 138,091 verified
reasoning trajectories.
Reflection-Enhanced Reasoning Sampling.Standard Re-
jection Sampling (RS) is inefficient as it discards the vast ma-
jority of trajectories where the teacher model fails ( ˆDexcludes
d∗). To address this, we propose RERS forWeak-to-Strong
generalization. LetM Tbe the teacher model (e.g., DeepSeek-
R1). For a clinical noteC, the initial inference process yields a
reasoning chain and diagnosis:(r, ˆD) =F MT(C). We define
the reasoning generation process as a conditional mapping to
construct a high-quality COT datasetR:R=(
(r,ˆD)if match( ˆD, d∗)
(rref,ˆDref)if¬match( ˆD, d∗)(1)
where(r ref,ˆDref) =F MT(C, r fail,K,E)denotes the
reflection-enhanced trajectory.
The first case represents standard positive sampling where
the reasoning successfully leads to the correct diagnosis. The
second case denotes our Reflection Mechanism: given a failed
reasoning pathr failwhere the correct diagnosis was missed,
we prompt the model to self-correct by synthesizing retrieved
knowledgeKand expert feedbackE(insights from other
models). This forces the processF MTto identify logical
fallacies inr failand internalize external knowledge to derive
d∗, effectively turning ”failure” into a dense training signal.
Trajectory Refinement.To ensure robustness and diversity,
we apply two augmentation strategies to the generation ofT:
1)Random Information Masking:We generate ˜C=
Mask(C)by randomly obscuring portions of the input,
compelling the model to reason with partial evidence.
2)Knowledge-Constrained Filtering:To eliminate process
hallucination, every generated trajectoryr∈ Tis post-
verified against factual triplets extracted fromK. Only
trajectories satisfying logical consistency and factual
accuracy are retained.
E. Dual-Level Curriculum Reinforcement Learning (DCRL)
Following the injection of knowledge and reasoning via
SFT, to move beyond static data dependence and achieve

autonomous evolution, we propose the DCRL algorithm.
Building upon the DAPO framework [23], we designed a
progressive curriculum from two distinct dimensions, enabling
the model to autonomously optimize its reasoning paths.
a) Dual-Level Curriculum Design:Unlike mathematical
reasoning, where errors primarily stem from insufficient rea-
soning capabilities, diagnostic failures can also result from a
lack of domain knowledge. To decouple and address these
challenges, we designed a curriculum operating at two levels:
•Task-level:Progresses fromDifferential Diagnosis(gen-
erating candidates or diagnosing with cues) toFinal
Diagnosis(directly determining a definitive result).
•Case-level:Samples are stratified into four tiers based on
preliminary diagnostic accuracy.
This structure enables targeted optimization: combinations of
Simple Cases with Hard Tasks focus on mastering multi-step
reasoning paradigms, while Hard Cases with Simple Tasks
prioritize the recall of disease-specific knowledge.
b) Scheduling Strategy:The training curriculum is de-
signed to progressively escalate in both case-level and task-
level difficulty. To mitigate catastrophic forgetting, we inter-
leave the current training subset with a small proportion of
samples from other difficulty tiers.
c) Early-Stage Filtering:Given the severe data scarcity
inherent to rare diseases—where many diseases have only
one record—we reused RareArena for RL stage. To maximize
training efficiency, we filtered out samples that achieved 100%
accuracy during SFT. Our preliminary experiments revealed
that reinforcement learning on these data yields negligible
performance gains and may even compromise performance.
F . Collaborative Reasoning Refinement (CRR)
While RareDxR1 achieves SOTA performance, single mod-
els inevitably exhibit biases. To leverage the complementarity
between models (Fig. 3), we introduceCRR, an inference-
time framework mirroring the RERS reflection mechanism.
LetMbe the trained RareDxR1 model. Given a clinical
noteC, the process proceeds in steps:
1. Initial Diagnosis.It generates an initial reasoning chain
rinitand a candidate set via(r init,ˆDinit) =F M(C).
2. Multi-Source Evidence Aggregation.We construct a
composite evidence set consisting of external peer opinions
and factual verification. LetM extbe a panel of external
models. We gather external diagnosesE={F Mk(C)|M k∈
Mext}and retrieve knowledgeKrelevant to the union of
candidate diseases.
3. Refined Inference.Finally, RareDxR1 acts as the lead
diagnostician, synthesizing the initial hypothesis with the
supplementary evidence to generate the final diagnosis:
(rfinal,ˆDfinal) =F M(C, r init,E,K).(2)
Crucially, this step utilizes the capability cultivated via
RERS—specifically, the ability to critique initial reasoning
(rinit) by weighing conflicting external evidence (E) and
factual constraints (K) to correct biases.IV. EXPERIMENTS
A. Experiment Settings
Our approach was implemented on two general reasoning
models, Qwen3-14B [31] and DeepSeek-Distill-14B [16], re-
sulting in RareDxR1-Q and RareDxR1-D. This dual-model
setup serves to both validate the general applicability of our
method and provide high-quality collaborators for the CRR
stage.
Our evaluation of RareDxR1 is threefold. First, we assess its
performance on our primary task, Rare Disease Diagnosis from
Unstructured Clinical Notes, using theRareArena-Test(1,025
cases, 637 diseases) [29] as the in-distribution benchmark
andMIMIC-IV-Rare(590 cases, 239 diseases) created from
MIMIC-IV-Notes [30] for OOD testing. Second, we conduct
extensive ablation studies to validate the contribution of each
of our proposed modules. Finally, we test generalization on
common diseases onDiagnosisArena[32] and diagnosis from
structured phenotype inputs onRareBench-Public[33].
Our comparative analysis included mainstream high-
performance models spanning open-source and closed-source
domains, covering both general-purpose LLMs [17], [34] and
medical reasoning LLMs [12], [35].
B. Main Results
Table I presents the results for rare disease diagnosis from
unstructured clinical notes. Following the two-stage training,
our RareDxR1-14B model achieves best performance across
all metrics, outperforming numerous models with significantly
larger parameter counts. A key indicator of its generalization
ability is its superior performance on the MIMIC-IV dataset,
despite this data being entirely unseen during training. This ro-
bust OOD performance suggests that our approach endows the
model with genuine, transferable diagnostic skills applicable
across a wide spectrum of rare diseases.
As shown in Table II, to validate whether RareDxR1 ac-
quires generalizable reasoning rather than rote memorization,
we stratified the test cases by the frequency of disease in the
train set. RareDxR1-Q consistently outperforms the DeepSeek-
R1-671B baseline. Crucially, in theZero-shotsetting, our
model achieves a50.00%Top-10 Recall (vs. 28.85% for
baseline). This confirms that RareDxR1 has internalized the
meta-logic of differential diagnosis, enabling it to infer correct
diagnoses even without prior exposure to specific disease
samples.
To investigate the theoretical basis for CRR’s effectiveness,
we analyzed the collaborative potential of multiple models
as shown in Fig. 3. This confirms the existence of comple-
mentary diagnostic strengths among models and validates that
synthesizing diverse perspectives, as CRR does, is a promising
direction for enhancing diagnostic accuracy.
C. Ablation Studies
To validate the contribution of each proposed component,
we conducted an ablation study, with results presented in Ta-
ble III. The findings reveal a clear, step-wise performance gain,
confirming the efficacy of our multi-stage training strategy.

TABLE I
EVALUATION ON RARE DISEASE DIAGNOSIS FROM UNSTRUCTURED CLINICAL NOTES. WE COMPARE THE PERFORMANCE OF BASELINE MODELS,OUR
SINGLE MODELS(RAREDXR1),AND OUR COLLABORATIVE REFINED MODELS(+CRR).
Model Params Domain RLMRareArena-Test (ID) MIMIC-IV-Rare (OOD)
Top-1 Top-3 Top-10 Top-1 Top-3 Top-10
GPT-4o - General No 41.66 50.54 56.88 29.15 33.39 37.97
Doubao-1.5-Thinking-Pro - General Yes 43.71 52.88 56.78 44.58 50.34 54.75
Gemini-2.5-Flash - General Yes 44.88 53.76 57.37 38.14 45.25 48.98
DeepSeek-R1 671B General Yes 44.20 54.73 59.02 42.71 48.64 54.58
DeepSeek-Distill 14B General Yes 25.76 35.12 42.15 39.83 45.76 48.31
Qwen3 14B General Yes 32.68 40.59 45.95 40.34 45.08 50.17
Baichuan-M1 14B Medical Yes 31.90 40.98 48.00 45.42 51.86 56.44
HuatuoGPT-o1 70B Medical Yes 36.29 41.27 41.76 34.24 37.97 44.07
Ours: RareDxR1 (Single Models)
RareDxR1-DeepSeek 14B Medical Yes 52.29 63.61 69.27 47.63 55.25 61.19
RareDxR1-Qwen 14B Medical Yes55.46 66.34 75.32 48.98 57.80 65.25
Ours: RareDxR1 + Collaborative Reasoning Refinement (CRR)
RareDxR1-DeepSeek + CRR 14B Medical Yes 56.98 69.56 75.41 52.03 60.34 65.59
RareDxR1-Qwen + CRR 14B Medical Yes60.29 71.71 78.73 53.05 62.20 69.83
TABLE II
TOP-1/10 RECALLBY DISEASE FREQUENCIES IN TRAIN SET. ”FREQ=0”
DENOTES UNSEEN DISEASES.
Group (Train-Freq) #CaseDS-R1-671B Ours-Q (14B)
Top-1 Top-10 Top-1 Top-10
Zero-shot (0) 52 19.23 28.8528.85 50.00
Few-shot (<19) 544 40.99 56.6243.38 67.46
Many-shot (≥19) 429 51.28 65.7369.23 88.34
1 2 3 4 5 6 7 8
Number of Collaborators (K)757677787980Top-10 Recall (%)
Best Single
(RareDxR1-Q)+RareDxR1-Q+RareDxR1-D+Gemini+Baichuan-M1
+Huatuo-O1+GPT-4o
+DeepSeek-R1
+Doubao(a) RareArena-Test
1 2 3 4 5 6 7 8
Number of Collaborators (K)6566676869
Best Single
(RareDxR1-Q)+RareDxR1-Q+Doubao
+Baichuan-M1+RareDxR1-D
+Huatuo+GPT-4o
+Gemini+DeepSeek-R1(b) MIMIC-IV-Rare
With Knowledge Base No Knowledge Base Best Single (RareDxR1-Q)
Fig. 3.Analysis of CRR Performance.Performance of our method as
collaborators are added with or without retrieved knowledge.K= 1
represents the model’s self-refinement using retrieved knowledge.
The most significant performance leap originates from our SFT
stage, which demonstrates the critical importance of domain-
specific adaptation. Within this stage, the inclusion of data
from our Reflection-Enhanced Reasoning Sampling pipeline
provides a distinct advantage, validating our hypothesis that
learning from corrected errors enhances reasoning quality.
Subsequently, applying Reinforcement Learning with our dual-
level curriculum further hones the model’s capabilities, deliv-
ering the final and most substantial boost in performance.
Notably, a ”weak-to-strong” generalization phenomenon
was observed. Even though powerful models like DeepSeek-TABLE III
ABLATION STUDY ON THE CONTRIBUTIONS OFSFTANDRLON THE
RAREARENA-TEST.
Model Configuration Top-1 Top-3 Top-10
(A) Base Model (DeepSeek-Distill) 25.76 35.12 42.15
+SFT Only
(B) SFT (w/o RERS Data) 45.85 56.20 60.78
(C) SFT (w/ RERS Data) 47.12 56.98 62.15
+RL
(D) RL Only 30.24 43.02 52.10
(E) SFT + RL (w/o Curriculum) 49.56 60.49 65.56
(F)SFT + RL (w/ Curriculum) 52.29 63.61 69.27
R1-671B served as teachers in the generation of our RareDx-
COT, our model, after only SFT, substantially surpassed its
teachers’ performance. We attribute this significant improve-
ment to our data generation strategies, such as knowledge-
constrained filtering and reflection-enhanced sampling, which
effectively distill and refine the teacher’s capabilities.
D. Cross-Domain Generalization Ability
To validate that our training cultivates a truly generalizable
medical reasoning ability, we evaluated RareDxR1’s zero-shot
performance on two distinct auxiliary tasks: Rare Disease
Diagnosis from Structured Phenotypes and Common Disease
Diagnosis from Unstructured Notes. The consolidated results
are presented in Table IV.
On RareBench-Public, RareDxR1 demonstrates robust
cross-modal transfer without phenotype training. Despite op-
erating in a more challenging open-domain setting compared
to PhenoBrain’s closed-set paradigm, it outperforms this spe-
cialized SOTA and rivals top-tier LLMs.
Furthermore, results on the DiagnosisArena benchmark
demonstrate that RareDxR1 generalizes effectively to common
diseases without suffering from catastrophic forgetting. No-

TABLE IV
ZERO-SHOT ACCURACY OFRAREDXR1ON AUXILIARY TASKS,
DEMONSTRATING STRONG GENERALIZATION ACROSS DIFFERENT DATA
MODALITIES(NOTES VS.PHENOTYPES)AND DISEASE DOMAINS
ModelRareBench-Public DiagnosisArena
Phenotypes, Rare Notes, Common
Specialized Method (Closed-set)
PhenoBrain 30.40 –
Large-scale LLMs (Open-domain)
DeepSeek-R1-671B37.92 45.57
Gemini-2.5-Flash 35.51 37.48
Doubao-1.5-Thinking-Pro 31.96 30.93
GPT-4o 30.01 24.04
Comparable-sized LLMs (Open-domain)
Baichuan-M1-14B 30.01 22.30
HuatuoGPT-o1-70B 29.44 18.69
DeepSeek-Distill-14B 30.93 27.76
RareDxR1-D-14B (Ours) 37.00 35.73
tably, it surpasses its base model (DeepSeek-Distill), indicating
that our training strategy enhances diagnostic capabilities.
V. CONCLUSION
RareDxR1 enables reasoning-centric rare disease diagno-
sis from unstructured notes. Through knowledge-reasoning
dataset construction and dual-level curriculum reinforcement
learning, our model outperforms larger baselines. Further-
more, our Collaborative Reasoning Refinement facilitates ef-
fective joint reasoning with external experts, underscoring
RareDxR1’s potential as a powerful tool for clinicians. Mean-
while, rigorous clinical validation and human oversight remain
essential to ensure patient safety in real-world applications.
REFERENCES
[1] M. Haendel, N. Vasilevsky, D. Unni, C. Bologa, N. Harris, H. Rehm
et al., “How many rare diseases are there?”Nature reviews drug
discovery, vol. 19, no. 2, pp. 77–78, 2020.
[2] Z. Yicheng and S. Zhang, “Retrospective on establishing rare diseases
medical service system and research platform in china,”Journal of Rare
Diseases, pp. 93–96, 2022.
[3] W. R. Evans, “Dare to think rare: diagnostic delay and rare diseases,”
The British Journal of General Practice, vol. 68, no. 670, p. 224, 2018.
[4] B. Martinez-Delgado and M. J. Barrero, “Epigenomic approaches for
the diagnosis of rare diseases,”Epigenomes, vol. 6, no. 3, p. 21, 2022.
[5] X. Mao, Y . Huang, Y . Jin, L. Wang, X. Chen, H. Liuet al., “A
phenotype-based ai pipeline outperforms human experts in differentially
diagnosing rare diseases using ehrs,”npj Digital Medicine, vol. 8, no. 1,
p. 68, 2025.
[6] W. Zhai, X. Huang, N. Shen, and S. Zhu, “Phen2disease: a phenotype-
driven model for disease and gene prioritization by bidirectional max-
imum matching semantic similarities,”Briefings in Bioinformatics,
vol. 24, no. 4, p. bbad172, 2023.
[7] J. Peng, H. Xue, Y . Shao, X. Shang, Y . Wang, and J. Chen, “Measuring
phenotype semantic similarity using human phenotype ontology,” in
2016 IEEE International Conference on Bioinformatics and Biomedicine
(BIBM). IEEE, 2016, pp. 763–766.
[8] M. Pinol, R. Alves, I. Teixido, J. Mateo, F. Solsona, and E. Vilapriny ´o,
“Rare disease discovery: An optimized disease ranking system,”IEEE
Transactions on Industrial Informatics, vol. 13, no. 3, pp. 1184–1192,
2017.[9] M. Zhao, J. M. Havrilla, L. Fang, Y . Chen, J. Peng, C. Liuet al.,
“Phen2gene: rapid phenotype-driven gene prioritization for rare dis-
eases,”NAR genomics and Bioinformatics, vol. 2, no. 2, p. lqaa032,
2020.
[10] C. A. Deisseroth, J. Birgmeier, E. E. Bodle, J. N. Kohler, D. R. Matalon,
Y . Nazarenko, C. A. Genetti, C. A. Brownstein, K. Schmitz-Abe,
K. Schochet al., “Clinphen extracts and prioritizes patient phenotypes
directly from medical records to expedite genetic disease diagnosis,”
Genetics in Medicine, vol. 21, no. 7, pp. 1585–1593, 2019.
[11] D. Smedley and P. N. Robinson, “Phenotype-driven strategies for exome
prioritization of human mendelian disease genes,”Genome medicine,
vol. 7, no. 1, p. 81, 2015.
[12] B. Wang, H. Zhao, H. Zhou, L. Song, M. Xu, W. Cheng, X. Zeng,
Y . Zhang, Y . Huo, Z. Wanget al., “Baichuan-m1: Pushing the medical
capability of large language models,”arXiv preprint arXiv:2502.12671,
2025.
[13] H. Wang, C. Liu, N. Xi, Z. Qiang, S. Zhao, B. Qin, and T. Liu, “Huatuo:
Tuning llama model with chinese medical knowledge,”arXiv preprint
arXiv:2304.06975, 2023.
[14] P. Talapova, M. Gargano, N. Matentzoglu, B. Coleman, E. Addo-Lartey,
A. Anagnostopouloset al., “The human phenotype ontology in 2024:
phenotypes around the world.” 2023.
[15] J. S. Amberger, C. A. Bocchini, F. Schiettecatte, A. F. Scott, and
A. Hamosh, “Omim. org: Online mendelian inheritance in man
(omim®), an online catalog of human genes and genetic disorders,”
Nucleic acids research, vol. 43, no. D1, pp. D789–D798, 2015.
[16] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhuet al., “Deepseek-
r1 incentivizes reasoning in llms through reinforcement learning,”Na-
ture, vol. 645, no. 8081, pp. 633–638, 2025.
[17] Google DeepMind, “Gemini pro,” 2025, accessed: 2025-07-26. [Online].
Available: https://deepmind.google/models/gemini/pro/
[18] J. Ahn, R. Verma, R. Lou, D. Liu, R. Zhang, and W. Yin, “Large lan-
guage models for mathematical reasoning: Progresses and challenges,”
arXiv preprint arXiv:2402.00157, 2024.
[19] D. Wu, Z. Wang, Q. Nguyen, and K. Wang, “Integrating chain-
of-thought and retrieval augmented generation enhances rare disease
diagnosis from clinical notes,” 2025. [Online]. Available: https:
//arxiv.org/abs/2503.12286
[20] X. Chen, Y . Jin, X. Mao, L. Wang, S. Zhang, and T. Chen, “Rareagents:
Advancing rare disease care through llm-empowered multi-disciplinary
team,” 2025. [Online]. Available: https://arxiv.org/abs/2412.12475
[21] W. Zhao, C. Wu, Y . Fan, X. Zhang, P. Qiu, Y . Sunet al., “An
agentic system for rare disease diagnosis with traceable reasoning,”
arXiv preprint arXiv:2506.20430, 2025.
[22] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang,
Y . K. Li, Y . Wu, and D. Guo, “Deepseekmath: Pushing the limits
of mathematical reasoning in open language models,” 2024. [Online].
Available: https://arxiv.org/abs/2402.03300
[23] Q. Yu, Z. Zhang, R. Zhu, Y . Yuan, X. Zuo, Y . Yueet al., “Dapo:
An open-source llm reinforcement learning system at scale,” 2025.
[Online]. Available: https://arxiv.org/abs/2503.14476
[24] Y . Lai, J. Zhong, M. Li, S. Zhao, and X. Yang, “Med-r1: Reinforcement
learning for generalizable medical reasoning in vision-language
models,” 2025. [Online]. Available: https://arxiv.org/abs/2503.13939
[25] S. Zhang, Q. Liu, G. Qin, T. Naumann, and H. Poon, “Med-rlvr:
Emerging medical reasoning from a 3b base model via reinforcement
learning,” 2025. [Online]. Available: https://arxiv.org/abs/2502.19655
[26] S. S. Weinreich, R. Mangon, J. Sikkens, M. E. Teeuw, and M. Cornel,
“Orphanet: a european database for rare diseases,”Nederlands tijdschrift
voor geneeskunde, vol. 152, no. 9, pp. 518–519, 2008.
[27] W. H. Organization,The ICD-10 classification of mental and be-
havioural disorders: clinical descriptions and diagnostic guidelines.
World Health Organization, 1992, vol. 1.
[28] K. Canese and S. Weis, “Pubmed: the bibliographic database,”The NCBI
handbook, vol. 2, no. 1, 2013.
[29] “Rarearena,” https://github.com/zhao-zy15/RareArena, 2024, accessed:
2025-05-01.
[30] A. E. Johnson, L. Bulgarelli, L. Shen, A. Gayles, A. Shammout,
S. Hornget al., “Mimic-iv, a freely accessible electronic health record
dataset,”Scientific data, vol. 10, no. 1, p. 1, 2023.
[31] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao,
C. Huang, C. Lvet al., “Qwen3 technical report,”arXiv preprint
arXiv:2505.09388, 2025.

[32] Y . Zhu, Z. Huang, L. Mu, Y . Huang, W. Nie, J. Liuet al., “Diagnosis-
arena: Benchmarking diagnostic reasoning for large language models,”
arXiv preprint arXiv:2505.14107, 2025.
[33] X. Chen, X. Mao, Q. Guo, L. Wang, S. Zhang, and T. Chen, “Rarebench:
can llms serve as rare diseases specialists?” inProceedings of the 30th
ACM SIGKDD conference on knowledge discovery and data mining,
2024, pp. 4850–4861.
[34] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkatet al., “Gpt-4
technical report,”arXiv preprint arXiv:2303.08774, 2023.
[35] J. Chen, Z. Cai, K. Ji, X. Wang, W. Liu, R. Wanget al., “Huatuogpt-
o1, towards medical complex reasoning with llms,”arXiv preprint
arXiv:2412.18925, 2024.