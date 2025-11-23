# KRAL: Knowledge and Reasoning Augmented Learning for LLM-assisted Clinical Antimicrobial Therapy

**Authors**: Zhe Li, Yehan Qiu, Yujie Chen, Xiang Zhou

**Published**: 2025-11-20 02:04:46

**PDF URL**: [https://arxiv.org/pdf/2511.15974v1](https://arxiv.org/pdf/2511.15974v1)

## Abstract
Clinical antimicrobial therapy requires the dynamic integration of pathogen profiles, host factors, pharmacological properties of antimicrobials, and the severity of infection.This complexity imposes fundamental limitations on the applicability of Large Language Models (LLMs) in high-stakes clinical decision-making including knowledge gaps, data privacy concerns, high deployment costs, and limited reasoning capabilities. To address these challenges, we propose KRAL (Knowledge and Reasoning Augmented Learning), a low-cost, scalable, privacy-preserving paradigm that leverages teacher-model reasoning to automatically distill knowledge and reasoning trajectories via answer-to-question reverse generation, employs heuristic learning for semi-supervised data augmentation (reducing manual annotation requirements by approximately 80%), and utilizes agentic reinforcement learning to jointly enhance medical knowledge and reasoning while optimizing computational and memory efficiency. A hierarchical evaluation employing diverse teacher-model proxies reduces assessment costs, while modular interface design facilitates seamless system updates. Experimental results demonstrate that KRAL significantly outperforms traditional Retrieval-Augmented Generation (RAG) and Supervised Fine-Tuning (SFT) methods. It improves knowledge question-answering capability (Accuracy@1 on the external open-source benchmark MEDQA increased by 1.8% vs. SFT and 3.6% vs. RAG) and reasoning capability (Pass@1 on the external benchmark PUMCH Antimicrobial increased by 27% vs. SFT and 27.2% vs. RAG), achieved at ~20% of SFT's long-term training costs. This establishes KRAL as an effective solution for enhancing local LLMs' clinical diagnostic capabilities, enabling low-cost, high-safety deployment in complex medical decision support.

## Full Text


<!-- PDF content starts -->

KRAL:Knowledge andReasoningAugmented
Learning for LLM-assisted Clinical Antimicrobial
Therapy
Zhe Lia,1, Yehan Qiub,1, Yujie Chenc,1, Xin Dingb, Yanwen Fub, Jing Sunb,
Yaguang Lia, Xinping Zhanga, Xiangyang Yea, Jieqing Chena, Wei Pana,
Yuna Weia, Chao Donga, Ziyang Huanga, Huizhen Jianga, Lian Maa,
Dandan Maa, Xiang Zhoua,b,∗
aInformation Center, State Key Laboratory of Complex Severe and Rare Diseases,
Peking Union Medical College Hospital, Beijing, 100730, China
bDepartment of Critical Care Medicine, State Key Laboratory of Complex Severe and
Rare Diseases, Peking Union Medical College Hospital, Peking Union Medical College
and Chinese Academy of Medical Sciences, Beijing, 100730, China
cMedical Intensive Care Unit, State Key Laboratory of Complex Severe and Rare
Diseases, Peking Union Medical College Hospital, Peking Union Medical College and
Chinese Academy of Medical Sciences, Beijing, 100730, China
Abstract
Clinical antimicrobial therapy requires the dynamic integration of pathogen
profiles, host factors, pharmacological properties of antimicrobials, and the
severity of infection.This complexity imposes fundamental limitations on
the applicability of Large Language Models (LLMs) in high-stakes clinical
decision-making including knowledge gaps, data privacy concerns, high de-
ployment costs, and limited reasoning capabilities. To address these chal-
lenges, we propose KRAL (Knowledge and Reasoning Augmented Learn-
ing), alow-cost, scalable, privacy-preservingparadigmthatleveragesteacher-
model reasoning to automatically distill knowledge and reasoning trajec-
tories via answer-to-question reverse generation, employs heuristic learning
for semi-supervised data augmentation (reducing manual annotation require-
ments by approximately 80%), and utilizes agentic reinforcement learning to
∗Corresponding authors
Email address:zx_pumc@163.com(Xiang Zhou)
1This is the first author footnote.arXiv:2511.15974v1  [cs.AI]  20 Nov 2025

jointly enhance medical knowledge and reasoning while optimizing compu-
tational and memory efficiency. A hierarchical evaluation employing diverse
teacher-model proxies reduces assessment costs, while modular interface de-
sign facilitates seamless system updates. Experimental results demonstrate
that KRAL significantly outperforms traditional Retrieval-Augmented Gen-
eration (RAG) and Supervised Fine-Tuning (SFT) methods. It improves
knowledge question-answering capability (Accuracy@1 on the external open-
source benchmark MEDQA increased by 1.8% vs. SFT and 3.6% vs. RAG)
and reasoning capability (Pass@1 on the external benchmark PUMCH An-
timicrobial increased by 27% vs. SFT and 27.2% vs. RAG), achieved at 20%
of SFT’s long-term training costs. This establishes KRAL as an effective so-
lution for enhancing local LLMs’ clinical diagnostic capabilities, enabling
low-cost, high-safety deployment in complex medical decision support.
Keywords:LLM, Antimicrobial Therapy, KRAL, Agentic Reinforcement
Learning, Resource-limited Healthcare
1. Introduction
Antimicrobial therapy constitutes a cornerstone of modern clinical prac-
tice. The formulation of an effective regimen necessitates the integration
of pathogen-specific factors, host characteristics, pharmacokinetic pharma-
codynamic (PK/PD) properties of antimicrobials, and infection severity, all
of which are dynamic and interrelated. This places significant cognitive load
on clinicians, especially for non-infectious disease specialists or in situations
where pathogens are unknown and time is limited, which may result in sub-
optimal prescribing decisions, thereby increasing the likelihood of therapeu-
tic failure, antimicrobial toxicity, and the emergence of multidrug-resistant
(MDR) pathogens. Large language models (LLMs) have recently emerged
as promising tools for enhancing clinical decision support systems (CDSS),
owing to their advanced natural language understanding and generation ca-
pabilities. Nevertheless, the direct deployment of general-purpose LLMs in
high-stakes clinical domains such as antimicrobial therapy is fraught with
limitations, including:
•Knowledge bias: Medical content constitutes <0.3 % of the pre-training
corpora[1] in mainstream LLMs (e.g., GPT-3), resulting in limited cover-
age of rare or emerging pathogens[2–4], outdated guideline adherence[5–
9](Appendix A1), and suboptimal performance on atypical presentations.
2

•Data privacy and compliance risks: The use of closed-source, cloud-
based LLMs (e.g., GPT-4) for processing unencrypted protected health
information(PHI)mayviolateHIPAA/GDPR-equivalentregulations,even
underprivatedeploymentscenariosifonlineguidelineupdatesarerequired[10–
12].
•Highdeploymentcosts: Primaryhealthcareinstitutionsfacedualshort-
ages of computing power and data. High-quality annotation relies on
clinical experts, which consumes significant time and resources [13], par-
ticularly for medical-specific large models. Existing IT infrastructure is
designed for traditional systems like HIS, lacking high-performance com-
putingclustersforlargemodels[14]. Upgradecoststhusrepresentacritical
bottleneck for technological implementation [15, 16].
•Reasoningbias: LLMsarepredominantlypre-trainedonstatic,guideline-
derived corpora (e.g., PubMed, MedQA), which lack real-world clinical
volatility,multi-stepreasoning,andpatient-specificcontextualization,lead-
ing to poor generalization in complex comorbid scenarios [17–19].
Among these, medical knowledge bias and reasoning bias are the most crit-
ical. Retrospective analysis of de-identified records from the open-source
MIMIC-IV database and institutional EMRs at PUMCH (Appendix A2)
indicates that in approximately 75% of antimicrobial prescribing scenarios,
cliniciansmustintegrateup-to-dateguidelineknowledgewithmulti-stepclin-
ical reasoning to arrive at appropriate therapeutic decisions. Consider, for
example, a patient with hypertension, type 2 diabetes mellitus, chronic kid-
ney disease (CKD), and a documented history of inappropriate antibiotic
exposure who presents with community-acquired pneumonia (CAP). Select-
ing an "appropriate" antimicrobial regimen in this context requires:
•Comorbidity-adjusted prescribing: CKD may contraindicate renally
cleared agents (e.g., aminoglycosides), whereas diabetes may modulate in-
fection severity and wound-healing capacity.
•Resistance-risk assessment: Prior antibiotic misuse increases the prob-
ability of colonization with multidrug-resistant (MDR) organisms (e.g.,
ESBL-producingEnterobacteriaceae, MRSA),renderingstandardfirst-line
agents ineffective.
•Drug–drug interaction screening: Chronic medications (e.g., antihy-
pertensives, hypoglycaemics) must be reviewed to avoid clinically signifi-
cant interactions with newly prescribed antibiotics.
3

Effective management of such complex clinical scenarios necessitates LLMs
capable of multi-step, context-aware reasoning and real-time access to evolv-
ing clinical guidelines. Conventional RAG frameworks do not mitigate rea-
soning deficits, whereas SFT demands extensive, expert-level annotation and
stillfallsshortincomplex, multi-stepreasoning. Toaddresstheselimitations,
we propose KRAL (Knowledge and Reasoning Augmented Learning), a mod-
ular, cost-efficient training paradigm that jointly enhances domain knowl-
edge and clinical reasoning in LLMs. KRAL aims to improve the reliability,
safety, and clinical utility of AI-driven antimicrobial recommendations, while
remaining deployable in resource-constrained settings.
2. Methods
2.1. Datasets
TheKRALtrainingcorpuswasassembledfromthreeinstitutionalsources:
(i) PUMCH antimicrobial guidelines, (ii) antimicrobial Q&A pairs extracted
from the hospital CDSS, and (iii) de-identified electronic medical records
(EMRs). The guidelines comprise 750 pages of clinician-curated PDF/JPG
files covering indications, dosing, renal adjustments, and resistance patterns.
The CDSS subset contains 105 manually verified Q&A pairs generated dur-
ing real-world clinical consultations and subsequently validated by infectious-
disease specialists. Additionally, patient EMR data underwent inclusion &
exclusion screening (Appendix A3), using medical record covers, basic in-
formation, and laboratory/imaging data as context with final antimicrobial
choices serving as ground-truth labels, which was approved by the National
Health Commission of China, and a waiver of informed consent was re-
ceived from the Ethics Committee of Peking Union Medical College Hospital
(PUMCH, ethics number I-23PJ1416). This yielded 710 de-identified cases
for clinical-reasoning training; all PHI was removed under an IRB-approved
protocol. To evaluate the generalization capability of the KRAL learning
paradigm, two external evaluation datasets unrelated to the training data
were utilized: MedQA and the PUMCH Antimicrobial Benchmark. These
were applied to assess knowledge-enhanced learning and reasoning-enhanced
learning effectiveness, respectively. MedQA is a publicly available online
dataset compiled from professional medical board examinations. It contains
questions in three languages: English (12,723 questions), Simplified Chinese
(34,251 questions), and Traditional Chinese (14,123 questions). Each data
entry comprises three columns: Question, Options, and Evidence. A sample
4

dataset is provided in Appendix C3. An antimicrobial-focused subset was
selected from the English-language MedQA comprising four categories: An-
tifungal Drugs, Antifungal Medications, Antiviral Drugs, and Antiviral Med-
ications as shown in Fig. 10 (total n=56). The in-house PUMCH Antimi-
crobial Benchmark is an expert-curated, de-identified test set sampled from
prospectively collected EMRs; no overlap with training data. It is divided
by complexity level into three scenarios: prophylactic antibiotic therapy, rou-
tine antibiotic therapy, and complex antimicrobial therapy for drug-resistant
bacteria (total n=26). Data examples are provided in Appendix C3.
2.2. KRAL Pipeline Overview
The KRAL framework performs automated, multi-round knowledge-and-
reasoning distillation. Each round takes (i) the current training corpus and
(ii) a frozen teacher model as inputs, and outputs an updated student check-
point. After a cycle, updating the knowledge base or swapping in a stronger
teacher entails only uploading new documents and editing a YAML config;
all downstream steps are fully automated. The remaining processes auto-
matically rerun to regenerate the model checkpoint end-to-end. Utilizing an
adapter-based approach to updating model parameters enables flexible com-
bination of prior training results as plugins. These can be incorporated into
the base model weights for specific clinical reasoning scenarios.
The single-round learning process comprises three stages as shown in Fig-
ure 1: Data Distillation, Agentic Reinforcement Learning, and Multi-expert
Hierarchical Evaluation.
2.3. Stage 1: Data Distillation
In this stage, a small subset of real clinical data serves as a seed database.
We leverage DeepSeek-R1 (671 B) to distill structured guideline knowledge
and multi-step clinical reasoning chains into a lightweight student archi-
tecture. This stage yields high-quality, task-specific corpora suitable for
parameter-efficient fine-tuning of resource-constrained student models.
First, PUMCH clinical guidelines are processed through OCR structur-
ing, document segmentation, and vectorization to form a vector database.
Vectorization employs the open-source embedding model BGE-m3, which
boasts robust versatility, multilingual support, and multi-granularity pro-
cessing capabilities. This model offers broad application prospects in infor-
mation retrieval and natural language processing. The resulting vectors are
then indexed and optimized using the FAISS vector database. A caching
5

Figure 1: Overview of the KRAL paradigm
mechanism stores up to 1,000 recent search results to enhance retrieval effi-
ciency for repeated queries. Additionally, search results are reordered using a
hit-heat and timestamp-weighted sorting algorithm to optimize ranking.The
full re-rank weight is computed as in follow equation:
Rrank=w srs+w prp+w trt
6

Wherethesymbolw srepresentsthecosine-similarityweight,w ptheevidence-
frequency weight,w tthe temporal-recency weight, andr s,p,tthe correspond-
ing scores.
For each retrieval, when a knowledge chunk is matched, its hit count is up-
dated in the vector database’s metadata according to the following rule,
whereβis the hit count update coefficient. Hybrid retrieval significantly
improves recall accuracy (see NIH Test).
rp=clip(r p+βR rank,1)
To construct the Seed Database, we leverage LLM prompt engineering to
generateQ&Adatapairsintheanswer-to-questionmannerwithinputknowl-
edge chunks randomly sampled from the vector database. Additionally, 105
curated antimicrobial Q&A data points from the CDSS system are incor-
porated. The detailed workflow is illustrated in Figure 2. All data in the
Seed Database undergoes heuristic learning through few-shot prompting to
achieve 5 times query augmentation. Finally, by configuring high LLM rea-
soning diversity, the Teacher Model generates multiple reasoning outputs for
each query using RAG, yielding 10,138 Q&A pairs stored in the training
dataset. Sample data is provided in Appendix B1.
Second, to further enhance clinical reasoning, we adopt ReAct (Reason-
ing + Acting) to produce traceable, multi-step reasoning trajectories from
teacher outputs. By specifying multi-round "Reasoning-Action-Observation"
paths, it upgrades single-step outputs into verifiable, interactive, and error-
correctable multi-step chain-of-thought (CoT) reasoning, significantly boost-
ing the model’s reasoning ability [20]. Figure 3 illustrates this workflow.
The agent’s operational steps—each action or thought—along with the final
output constitute Reasoning Trajectory Data. Unlike standard SFT train-
ing data containing only query-answer pairs, reasoning trajectory data in-
corporates the reasoning process, enhancing stability in subsequent transfer
learning. Trajectory data examples are provided in Appendix B2.
2.4. Stage 2: Agentic Reinforcement Learning
The widespread application of the DeepSeek-R1 model demonstrates re-
inforcement learning’s significant enhancement of reasoning capabilities [21].
To simulate themulti-step decision-making characteristic of clinicaldiagnosis
while better suppressing hallucinations of critical information (e.g., medica-
tion dosage) under limited SFT training data, we employ an Agentic Rein-
forcement Learning approach in which agent interacts with a retrieval tool
7

Figure 2: Schematic of the data screening procedure
Figure 3: Reasoning trajectory distillation
and optimises a multi-turn policy via Group Relative Policy Optimisation
8

(GRPO). This enables online interaction between the LLM and retrieval
tools during training, with the Reward Model evaluating different actions
using distinct scoring logics. To address the cold start problem in knowledge
transferring [22], RL requires prior supervised fine-tuning based on Q&A
pairs. The training dataset comprises 30K entries, each consisting of input
(patient age, gender, chief complaint, medical history, and present illness)
and ground truth (actual clinical recommendations). The test set consti-
tutes 10% of the data, with examples provided in Appendix C1. Training
parameters are detailed in Table 1.
Referring to DeepSeek-R1’s implementation, we employs GRPO as the
reinforcement learning algorithm. Unlike PPO, GRPO eliminates the value
model by normalising rewards within a sampled group, cutting GPU memory
by 50 %. This significantly reduces hardware resource consumption during
RLtraining,makingGRPOmoresuitableforscenarioswithlimitedhardware
resources and sparse rewards in large models. Agentic GRPO was executed
on 1517 clinician-verified reasoning trajectories; hyper-parameters are listed
in Table 2.
Table 1: SFT Training Hyperparameters
Key Value Key Value
Learning rate 4e-5 Optimizer Adamw
Batch size 8 Gradient clipping 0.3
Training epochs 2000Pipeline
parallelism8
Adapter type LoRA/rank 32 Warm-up ratio 0.05
Gradient
accumulation
steps4 Deepspeed type Zero3+offload
2.4.1. Hardware-Efficient Implementation
Although GRPO already lowers memory, we further reduce computing
cost via LoRA (rank 16), FP8 mixed precision, ZeRO-3 sharding, CPU of-
floading and so on.
•LoRA(Low-rankAdaptation): Aparameter-efficientfine-tuningmethod
that trains only a small portion of the model’s parameters (specifically,
9

Table 2: GRPO training hyperparameters
Key Value Key Value
Learning rate 2e-6 Optimizer Adamw
Batch size 8 Gradient clipping 0.3
Training epochs 10Pipeline
parallelism4
Adapter type LoRA/rank 16 Warm-up ratio 0.1
Reward functionsCustom reward
functionsReward weights [1.0, 0.8]
Advantage clip
ratio-0.1/+0.4KL punishment
weight0.001
low-rank matrices) instead of the entire model, typically targeting just
0.1%-1% of parameters. When training only 1% of parameters, the com-
putational load for backward propagation gradients is reduced by approx-
imately 99% (theoretically reducing FLOPs proportionally). However, the
computational load for forward propagation remains largely unchanged, as
the base model is frozen with only minimal additional computation from
LoRA. Overall, computation is reduced by about 75%, and memory usage
is reduced by about 67% [23].
•FP8 mixed precision: FP8 mixed precision involves storing weights,
gradients, and activations using 8-bit floating-point numbers (FP8), while
employinghigherprecision(e.g., FP16/FP32)forcriticaloperations. Com-
pared to FP16, this approach reduces GPU memory usage by approxi-
mately 50%.
•Zero3[24]: Partitions optimizer states, gradients, and parameters across
multiple GPUs to eliminate redundancy. Each GPU stores only its own
slice, reducing memory usage in proportion to the number of GPUs (N).
•CPU offloading[25]: Offloading optimizer state, gradients, or parame-
ters to CPU memory or NVMe storage thereby frees up GPU memory.
With sufficient CPU memory, this can reduce GPU memory usage by ap-
proximately 87.5%.
•Compressed Reward Model: The number of models used for learn-
ing is reduced from two (Policy model + Value model) to one (Policy
model), resulting in a 50% reduction in computational resource consump-
tion. If GRPO’s reward model employs a lightweight model such as BGE-
10

m3 (0.6B), GPU memory usage decreases from four models to two, achiev-
ing a 50% reduction in GPU memory consumption.
2.4.2. Reasoning Augmented Implementation
To further augment reasoning capabilities, the following improvements
were implemented:
1.Asymmetric Clipping with Higher Upper Bound: The original
symmetric range[1−ϵ,1 +ϵ]is replaced with two hyper parameters
ϵlowandϵ high. The upper bound is relaxed to 0.28 or higher, while
the lower bound remains constrained between 0.1 and 0.2. This al-
lows model to learn ’key tokens’—those with low probability but high
advantage—efficiently.
2.Reward Smoothing: A custom attention-weighted reward function
incorporatingmedicalknowledgewasdeveloped. Thissmoothingmech-
anism filters out noisy contributions in model outputs, enhancing train-
ing stability and consistently improving reasoning capabilities.
2.4.3. Trajectory Data Preprocessing
Prior to GRPO training, trajectory data underwent targeted preprocess-
ing to enhance model robustness and learning efficiency. Specifically: First,
prompt influence was minimized to prevent excessive reliance on context dur-
ing reinforcement learning. Second, chat template adjustments preserved in-
ternal "thinking" tokens, making the reasoning process observable and learn-
able. Third, we reuse the teacher model with prompts to compress retrieved
knowledge, eliminating irrelevant information and reducing context length.
Finally, we streamlined multi-turn dialogues, retaining only the most concise
effective reasoning paths. These steps focused on refining two core capabil-
ities: (A) concise yet comprehensive analysis of medical records and condi-
tions; and (B) rational decision-making regarding when and how to retrieve
external knowledge.
2.4.4. Custom EMR-aware Reward
Toaligntherewardsignalwithpatient’sEMRdata, wedesignedacustom
reward.
1.Progress Reward: Given a patient’s record, the model must deter-
mine whether to perform a Retrieval Action and what keywords to pass
as search terms before making the final answer. An action sample is:
11

<action>keyword A, keyword B, keyword C... </action>. To enhance
evaluation robustness, Subword-level Jaccard similarity (rather than
traditional word set Jaccard[26]) is used to capture subword overlap,
e.g., between ’COVID’ and ’COVID-19’. Additionally, a max opera-
tor ensures each predicted term matches only the most similar ground
truth term, avoiding redundant penalties. Finally, the average similar-
ity across all predicted terms produces the final reward for the corre-
sponding action.
2.Subword-level Jaccard: For any two wordsw iandw j, the subword-
level Jaccard similarity is defined as:
Jaccard(w i, wj) =|C(w i)∩C(w j)|
|C(w i)∪C(w j)|
whereC(w)represents the deduplicated subword set ofw. When use
GTas collections of ground truth labelg,Pas collections of prediction
p, the final similarity (i.e., Reward) can be written as:
ActionReward=1
|P|X
p∈Pmax
g∈GTJaccard(p, g)
3.Hybrid Similarity: The final reward is defined as the similarity be-
tween the predicted and ground truth textual outputs. To enhance
robustness, we adopt a Hybrid Similarity metric based on embedding
vector distance. This approach builds upon direct textual embed-
ding comparisons by incorporating sparse bag-of-words similarity (lex-
ical matching score [27]) and semantic similarity (Colbert score [28]),
thereby better capturing key term and semantic similarities.
Hybrid(P, T) =1
NNX
i=1max
j
αS(i,j)
d+βS(i,j)
l+γS(i,j)
c
Subscriptsd, l, cdenote dense vector distance, lexical matching score,
and Col-BERT score respectively.i, jrepresent the chunk indices of
the prediction and ground truth respectively.
4.Repetition Penalty: To prevent reward hacking that degrades model
performance, a repetition penalty is introduced. To accommodate Chi-
nese grammatical peculiarities, Chinese part-of-speech tagging is in-
corporated, preventing erroneous expressions caused by repeated text
fragments with identical semantics.
12

5.Hybrid Similarity RewardThe final therapy-level reward quanti-
fies the semantic overlap between the generated regimenPand the
clinician-verified referenceT. We define a chunk-wise hybrid similar-
ity:
Rhybrid (P, T) =1
|P||P|X
i=1max
j∈[1,|T|]
α·S(i,j)
d|{z}
dense cosine+β·S(i,j)
l|{z}
lexical overlap+γ·S(i,j)
c|{z}
ColBERT
whereα+β+γ= 1, S(i,j)
d= cos 
h(i)
p,h(j)
t
is the dense embedding
cosine ;S(i,j)
lis the sparse lexical match computed with BM25 + uni-
gramoverlap[27];S(i,j)
cisthelate-interactionColBERTscore[28]using
medical-domain fine-tuned checkpoints. Chunks are sentence-level to
preserve dose-and-schedule boundaries.
6.RepetitionPenalty.Todiscouragesurface-formrewardhacking(e.g.,
repeateddrugnamesordosages), wepenalisesemantic-levelduplicates:
Rrep=−λKX
k=1⊮
cos 
hk,hk−1
> τ
·POS-weight(k),
where⊮[·]is the indicator function,τ= 0.92, and POS-weight down-
weights Chinese function-word repetitions (POS tagged by LACv2.1)
whilepreservingtherapeutic-contentrepeats(e.g.,“q8h” appearingtwice
for dual therapy is not penalised). The final token-level reward is
Rtoken =Rhybrid +Rrep.
2.5. Stage 3: Multi-expert Hierarchical Evaluation
Considering clinical implementation constraints, validation data primar-
ilyconsistsofde-identified,unstructuredmedicalrecordsandtreatmentplans
(sample data in Appendix C2). But traditional rule-based metrics (BLEU,
ROUGE) show weak correlation with clinical appropriateness and full hu-
man review is cost-prohibitive (approximate USD 200 per 100 cases). We
therefore implement a two-tier hierarchical evaluation protocol (Figure 4):
1.LLMsPre-review: Byconfiguringtemperature-scaledautoregression
and role-specific prompts, multiple expert avatars with distinct prefer-
ences are created. Each avatar produces a 5-point Likert score for
every therapy chunk; the final automated score is the median-of-five,
with standard deviation computed across avatars.
13

2.HumanStratifiedEvaluation: stratifiedsamplingbasedonstandard-
deviation quantiles of inter-avatar disagreement, yielding low, medium,
and high discordance strata.Human reviewers re-score a pre-specified
fraction within each stratum; if Cohen’s k < 0.8, the entire stratum is
re-sampled (max 3 rounds).Iteration terminates when k > 0.8 or 95 %
CI width < 5 % of the stratum mean. The full workflow is shown in
Figure 4.
Figure 4: Hierarchical Evaluation Pipeline
2.6. Generalization Evaluation
TovalidateKRAL’sgeneralizationcapability,evaluationexperimentswere
conducted on two held-out datasets: MedQA and PUMCH Antimicrobial.
MedQA tests factual antimicrobial knowledge; PUMCH Antimicrobial tests
multi-step clinical reasoning. Accuracy@1 (MedQA) and Pass@1 (PUMCH)
werepre-specifiedprimaryend-points; 95%CIsareClopper-Pearson; multiple-
comparison correction (Holm–Bonferroni) applied across 3 groups (RAG,
SFT, KRAL). Student model architecture: DeepSeek-R1-Distill-Qwen-32B
(32 B params); identical base model for RAG, SFT and KRAL to ensure fair
comparison. RAG and SFT methods were selected as control approaches for
knowledge augmentation and reasoning enhancement, respectively. Hyper-
parameters for RAG (Table 2), SFT (Table 3) and inference (Table 4) were
grid-searched on 20 % held-out development split; best median dev score was
locked before final test evaluation.
2.7. Ablation Study
To validate the effectiveness of the proposed algorithmic optimizations,
we quantify the marginal contribution of each component via ablation exper-
iments on the PUMCH Antimicrobial benchmark. Primary end-point: mean
token-level reward averaged over three random seeds (42, 123, 2024). Four
14

Table 3: RAG Setting
Key Value Key Value
Chunk size 256 tokens Dense weight 0.4
Chunk overlap 32 tokens Sparse weight 0.2
Embedding
modelBAAI/BGE-m3 Colbert weight 0.4
Search type Hybrid searchColbert
dimension1024
topk 3 Filter threshold 0.3
Table 4: Model Inference Setting
Key Value Key Value
GPUs 4*NVIDIA L20 Temperature 0.2
Inference Server vllm Think mode true
Tensor
Parallelism4 Max tokens 4096
GPU utilization 60% Top p 0.01
Max model
length20480 n 1
factors were systematically ablated as in Table 5. Experiments were per-
formed on the PUMCH benchmark with training parameters consistent with
Table 2 in the GRPO Training Process, utilizing computational resources
from 8 * NVIDIA L20 GPUs.
3. Results
KRAL simultaneously boosts antimicrobial knowledge (MedQA Accu-
racy@1 +1.8 % vs SFT, +3.6 % vs RAG) and clinical reasoning (PUMCH
Pass@1 +27 % vs SFT, +27.2 % vs RAG) while cutting compute cost by 8×
and VRAM by 100×, details are as follows.
3.1. Knowledge Retrieval Efficiency
We benchmark against BAAI’s official algorithm(FlagEmbedding) on 200
randomly sampled training queries using single NVIDIA L20[48 GB], batch
15

Table 5: Ablation Study Setting
Factor Description Ablated Setting
Clip-Higherasymmetric advantage
clipping[+0.4,−0.2]→ ±0.2
Reward Smoothingcustom EMR-aware
rewardhybrid similarity→
precise match
Sub-word Jaccard token-level matchsub-word→
word-level
Repetition Penalty surface-form penaltyλ= 0.1→0
size = 1, warm cache. After hybrid searching, re-ranking and tensor paral-
lelism optimization, mean latency dropped from 1.415 s→0.873 s (paired
t-test, p < 0.001), a 38 % reduction (Table 6).
Table 6: Needle-in-a-Haystack (NiH) Test
Dense Retrieval Hybrid Retrieval
Top@1 71.2% 76.6%
Top@3 85.9% 87.9%
Top@5 88.9% 90.7%
Table 7: Query Latency Test
BAAI Official Algorithm[29] Our Implementation
1.415s 0.873s
3.2. Supervised Fine-Tuning & GRPO Training Results
SFT achieved optimal validation accuracy of0.792±0.02(2800 steps,
early stopping). GRPO reached a peak validation reward of0.77±0.01after
12k training steps. Learning curves are smoothed with Exponential Moving
Average (β= 0.8) and shown in Figure 5 (SFT) and Figure 6 (GRPO).
16

Figure 5: SFT Process
3.3. Hardware Efficiency
Combined optimisations (LoRA-r16, FP8, ZeRO-3, CPU-offload, C.R.M)
yield:
FLOPs KRAL = 0.25 LoRA×0.5C.R.M = 0.125 (8×reduction)
VRAM KRAL = 0.33 LoRA×0.5FP8×0.125 offload×0.5C.R.M≈0.0103 (100×reduction)
Taking fine-tuning or RL with a 32B model as an example:
1.Theoretical full fine-tuning: 384 GB VRAM (64 params + 64 grad
+ 256 Adam).
2.Full RL (PPO): > 1.15 TB VRAM (3×models, at least 16 A100 or
H100 GPUs).
3.KRAL: 4×L20-48 GB for SFT, 8×L20-48 GB for GRPO (Table 6).
So KRAL can achieve high hardware efficiency on limited resource which
enables practical deployment.
17

Figure 6: GRPO Training Process
Table 8: Hardware Efficiency before & after optimization
Before Optimal After
Optimization
SFT 8×A100/H100
(80G)4×RTX4090/L20
(24–48G)
Expensive ($72k -
$240k)Cheap ($8k–$16k)
RL >16×A100/H100
(80G)8×L20 (48G)
Expensive
($14k–$48k)Cheap ($32k)
3.4. Generalization Evaluation
3.4.1. External open-source benchmark MedQA
Evaluation results demonstrate that the KRAL method achieves a +1.8%
accuracy improvement over SFT and a +3.6% improvement over RAG with
18

statistical significance mark ’*’ in Figure 7. This validates that the KRAL
method can effectively achieve knowledge augmentation even on external test
sets.
Figure 7: MedQA distribution & evaluation result
3.4.2. External proprietary PUMCH benchmark
Considering the unstructured nature of the data, the Pass@1 metric was
used to evaluate whether treatment plan keywords were accurately identi-
fied. Evaluation was conducted on approximately 100 data points. Results
demonstrate that KRAL significantly outperforms RAG and SFT across all
three subtasks marked by ’*’ in Figure 8. Overall Pass@1 improved by +27%
compared to SFT and +27.2% compared to RAG, validating the effectiveness
of enhanced medical reasoning capabilities in real clinical scenarios.
3.4.3. Comprehensive Comparison
The comprehensive comparison across both test datasets, incorporating
additional dimensions such as hardware utilization and data security, is pre-
sented in Fig. 9. Here, K.A. denotes Knowledge Augmentation, and R.A.
denotes Reasoning Augmentation. Their respective values are derived from
Accuracy@1 * 100 on the MedQA evaluation set and Pass@1 * 100 on the
19

Figure 8: Results on PUMCH benchmark
PUMCH Antimicrobial evaluation set. Training Cost represents the com-
bined cost of GPU resource allocation and training data annotation. Since
RAG does not involve training, it is excluded from comparison, with SFT
training cost serving as the baseline. Analysis of actual runtime testing and
Table 4 indicates that training the 32B student model KRAL requires at
least 8 L20 GPUs, while traditional full-parameter SFT demands at least 8
A100 GPUs. LoRA-efficient parameter tuning SFT requires at least 4 L20
GPUs. Training data comprises approximately 10K entries, with annota-
tion costs estimated at $2 per medical record (including diagnostic reasoning
chain). Teacher model API pricing per million tokens is negligible. Hardware
configuration and annotation cost comparisons are as follows:
1.Full-parameter SFT: > $72,000 + $20,000 = $92,000
2.LoRA SFT: $16,000 + ($2 * 10,000) = $36,000
3.KRAL: $32,000 + ($2 * 2,000) = $36,000
In the short term, KRAL costs are comparable to efficient SFT; in the long
term, as the labeling cost advantage (1:5 ratio) accumulates, hardware costs
are gradually amortized, ultimately requiring only about 20% of SFT’s ex-
20

penditure. Compared to traditional SFT, KRAL offers both cost and per-
formance advantages.
Figure 9: Results for K.A., R.A., Training Cost and Data Safety
3.5. Ablation Study
ResultsshowninFigure11demonstratethattheoptimizationproposedin
this study enable RL training with better performance and greater stability.
4. Discussion
The proposed KRAL paradigm achieves significant performance improve-
ments in clinical antimicrobial treatment decision-making through its dual-
enhancement framework for knowledge and reasoning capabilities. Experi-
mental results demonstrate that this framework addresses core bottlenecks in
traditional methods—such as medical knowledge gaps, data security, deploy-
mentcosts, andreasoninglimitations—whileprovidingafeasiblepathwayfor
deploying LLMs in resource-constrained healthcare settings. From a techno-
logical innovation perspective, KRAL’s three-stage learning framework (data
distillation, agent reinforcement learning, multi-expert evaluation) forms a
closed-loop optimization mechanism.
21

Figure 10: Results of the ablation experiments
During the data distillation phase, a vector knowledge base was con-
structed from PUMCH clinical guidelines, achieving optimized knowledge re-
trieval performance. Test results indicate that the proposed Hybrid Retrieval
approach reduces processing time by 38% compared to the official BAAI im-
plementation. We developed a heuristic medical data distillation technique,
leveragingknowledgechunkstoguideLLMsinreverse-generatingQ&Atrain-
ing data (answer-to-question generation). Employing a semi-supervised data
augmentation strategy, we utilized few-shot prompt engineering to guide
large language models in generating similar queries, expanding seed data
from 2k to 10k instances and effectively addressing the scarcity of medical
annotated data. Faced with massive unstructured clinical data, reasoning
teacher model(e.g., DeepSeek-R1) innovatively leverages its reasoning capa-
bility to manage and filter raw clinical data, drastically reducing manual
data curation costs. With screened data, DeepSeek-R1 employs its explicit
22

reasoning capability and agentic RAG technique to achieve comprehensive
documentation of clinical reasoning chains and guideline references.
During the training phase, we introduced Agentic RL for transfer training
in the medical domain. Agentic RL treats the student LLM as a complete
agent capable of long-term action, planning, and reflection within dynamic
environments. It employs RL for end-to-end optimization of the agent’s
multi-step decision policy, rather than merely refining the rationality of sin-
gle responses. Compared to traditional RLHF’s single-step decision-making,
Agentic RL explicitly extends the decision-making chain, making it highly
suitable for complex, multi-round clinical scenarios. These scenarios re-
quire iterative knowledge retrieval based on patients’ intricate medical histo-
ries, conditions, and drug interactions. Simultaneously, online invocation of
knowledge retrieval tools during training effectively mitigates hallucinations
of critical information—such as medication dosages—even with limited fine-
tuning data. Differences from traditional RL are detailed in Appendix C5.
Tobetteradapttomedicaltasks, acustomizedMedical-GRPOalgorithmwas
developed to enhance training stability, with ablation experiments validat-
ing its effectiveness. Notably, to overcome hardware efficiency bottlenecks, a
combined optimization scheme employing LoRA parameter fine-tuning, FP8
mixed-precision computation, ZeRO-3 parallel strategy, memory offloading,
and compressed reward models reduced computational power consumption
by 8-fold and memory usage by 100-fold. This enabled SFT training of
32B-parameter models on consumer-grade GPUs and reinforcement learning
training on 8 L20 GPUs. The evaluation phase established a hierarchical
multi-expert assessment system. By combining preliminary screening using
large models with stratified sampling by human experts, this system ensured
evaluation accuracy while controlling costs.
Evaluation experiments were conducted across two dimensions: the pub-
licly available MedQA test set and the proprietary PUMCH Antimicro-
bial QA test set. The MedQA set, a widely used benchmark derived from
professional medical examinations, assesses large models’ medical question-
answering capabilities. The PUMCH set, designed by Peking Union Med-
ical College Hospital experts, utilizes de-identified clinical data. Training
and evaluation datasets showed no overlap. Evaluation metrics included
Accuracy@1 and Pass@1. Results demonstrate that the KRAL framework
achievesa1.8%higherAccuracy@1thanbaselinemodel(DeepSeek-R1-Distill-
Qwen-32B) on MedQA, validating its knowledge enhancement and general-
ization. On the PUMCH Antimicrobial QA dataset, KRAL achieved an
23

80.8% Pass@1 rate, significantly outperforming traditional SFT (53.8%) and
RAG(53.6%). Thisdemonstratesthatdistillingreasoningpathsandincorpo-
rating agentic reinforcement learning substantially improve student models’
decision-making performance, especially in complex tasks like comorbidities
management and drug resistance risk assessment, validating the reasoning
enhancement effectiveness.
Compared to traditional approaches, the core advantages of KRAL include:
1.KnowledgeAugmentation-TraditionalSFTimpairsgeneralknowl-
edge retention, and excessive samples may even cause performance
degradation [30]. KRAL generates answers via heuristic data distil-
lation, leveraging both the knowledge base and the model’s pre-trained
general knowledge. This approach adapts techniques from SFT with
mixed pre-training corpora to minimize general knowledge loss while
avoiding the additional training costs associated with dataset expan-
sion.
2.ReasoningAugmentation-TheDeepSeek-R1technicalreportdemon-
strates reinforcement learning’s significant role in enhancing model
reasoning. Agentic reinforcement learning further integrates chain-
of-thought (CoT) techniques, utilizing step-by-step guidance to en-
sure continued improvement in reasoning abilities. Unlike traditional
fine-tuning, research indicates that many "medically fine-tuned" LLMs
strugglewithgeneralreasoningonuntrainedmedicaltasks, particularly
incomprehension,textgeneration,andencoding[31]. Incontrast,mod-
els trained with agentic reinforcement learning maintain strong gener-
alization on external evaluation datasets.
3.Hardware Efficiency- While traditional RAG offers low deployment
costs, it fails to enhance the model’s inherent reasoning capabilities and
cannot be directly applied in complex medical scenarios. Traditional
SFT incurs substantial computational/storage overhead and data re-
quirements [32]. For complex or long-context tasks, fine-tuning mod-
els necessitates extensive labeled data and domain-specific corpora to
achieve improvements [30]. KRAL constructs a multidimensional data
augmentation system through semi-supervised enhancement and LLM-
assistedmining, significantlyreducingmanualannotationdemandsand
costs. A fivefold data expansion achieves an 80% reduction in annota-
tion expenses. Furthermore, by combining efficient LoRA fine-tuning,
FP mixed-precision training, Zero3 model partitioning, and parame-
24

terunloadingoptimization, KRALsubstantiallyreducescomputational
resource consumption and VRAM usage during training. This en-
ables reinforcement fine-tuning of hundred-billion-parameter models on
consumer-grade GPUs.
4.Data Safety- The 100×VRAM reduction permits on-premise deploy-
ment within hospital firewalls, eliminating transmission of raw PHI to
external APIs and ensuring HIPAA/GDPR compliance by design—an
inherent advantage over cloud-based closed-source services (e.g., GPT-
5, Gemini).
5.Knowledge & Reasoning Scalability- Local deployment supports
push-buttonknowledgerefreshesviaPDF/CSVuploadsandhot-swapping
of teacher checkpoints (e.g., from DeepSeek-R1 to Med-Gemini). Fol-
lowing updates to either knowledge or the teacher model, the entire
process operates automatically, requiring only minimal human inter-
vention for sample verification. Furthermore, the same student model
can integrate multiple checkpoints from different iterations into one
latest checkpoint, enabling continuous augmentation.
Limitationsofthisstudyinclude: (1)Domainrestriction: guidelinescover
only antimicrobial therapy; oncology/cardiology generalization performance
remains to be proven. (2) Teacher bias: distilled reasoning trajectory in-
herit any heuristic or cognitive bias present in the teacher model, potentially
propagating systematic errors. (3) Sample size: evaluation cohort is mod-
est; multi-centre, longitudinal audits (>10 000 cases) are planned to confirm
long-term clinical benefit.
5. Conclusion
This study proposes a Knowledge and Reasoning Augmented Learning
(KRAL) paradigm to address four core challenges in applying large mod-
els to clinical antimicrobial therapy: medical knowledge bias, data security
risks, high deployment costs, and insufficient reasoning capabilities. Through
dual-source distillation of vectorized knowledge bases and teacher model rea-
soning trajectories, combined with an Agentic Reinforcement Learning strat-
egy, KRAL achieves efficient transfer of knowledge and reasoning capabilities
while maintaining HIPAA/GDPR compliance via on-premise deployment.
External evaluations show superior knowledge retention (MedQA +1.8 % vs
SFT) and clinical reasoning (PUMCH Pass@1 +27 % vs SFT) at 20 % of
25

long-term annotation cost and 100 times less VRAM.Multi-centre trials and
extension to oncology/cardiology guidelines will be pursued to confirm gen-
eralization performance. This paradigm offers a new approach for low-cost,
high-safety application of LLMs in complex clinical decision-making scenar-
ios, potentially accelerating the large-scale implementation of AI in precision
medicine.
6. Funding
This work was supported by the Beijing Municipal Natural Science Foun-
dation (L222019); CAMS Innovation Fund for Medical Sciences (CIFMS)
(2024-I2M- C&T-C-002); National High Level Hospital Clinical Research
Funding(2022-PUMCH-B-115&2022-PUMCH-D-005); andNationalKeyR&D
Program of China (2024YFF1207104).
7. Acknowledgments
Group authorship: China Critical Care Clinical Trials Group (CCCCTG)
and China National Critical Care Quality Control Center Group (China-
NCCQC group).
Appendix A.
Appendix A.1. Time Difference Between Mainstream Model Release Dates
and Pre-Training Knowledge Updates, Data Sources [4-8]
ModelCorpus Update
TimeRelease Date Gap(month) Data Source
GPT-4
Technical
ReportGPT-4 Turbo
(Apr 2024)12 arXiv:2303.08774 April 2024
GPT-4o System
CardGPT-4o (Oct
2023)7OpenAI, May
2024May 2024
Qwen2-VL
paperQwen2-VL (Jun
2023)20 arXiv:2409.12191 February 2025
DeepSeek-V2
paperDeepSeek-V2
(Dec 2023)5 arXiv:2405.04434 May 2024
Claude 3 Model
CardClaude 3 (Aug
2023)7Anthropic,
March 2024March 2024
26

Appendix A.2. MIMIC IV & Peking Union Medical College Hospital Inpa-
tient Antibiotic Usage Distribution
Dataset Inclusion and exclusion Size
MIMIC IVInpatient Orders & Order
Type: Antibiotics40687720→848482
PUMCHAdmitted in 2024 with
orders including
antibiotics, not
readmitted, and diagnoses
related to "infection,
pneumonia, enteritis,
typhoid fever, dysentery,
tuberculosis, viral
encephalitis." Order name
contains: "Penem" or
"Mycin" or "Cef" or
"Floxacin" or "Lover" or
"Coxazol" or "Piracilin"
or "Tigecycline" or
"Tecolaran" or
"Capreomycin" or
"Metronidazole" or
"Polymyxin" or
"Amikacin" or
"Sulfamethoxazole" or
"Linezolid" or Doxycycline
or Amoxicillin3999->710
27

Appendix B.
Appendix B.1. Knowledge Distillation Q&A pair sample
Q&A pair are as follows:
Question
"What are the symptoms of childhood chlamydia infection?"
Answer
Typical symptoms of chlamydia infection in children primarily include: 1.
No feverThe infection typically does not cause a fever (unlike bacterial in-
fections). 2.Staccato coughCharacterized by short, frequent dry coughs
resembling a "machine gun" cough. 3.Persistent coughIn older children
or adolescents, this may manifest as a cough lasting longer than 2 weeks
(requiring differentiation from pertussis or mycoplasma infection). 4.Lab-
oratory IndicatorsSerum IgM antibody titer >1:8 supports the diagnosis
of Chlamydia trachomatis infection.Additional Considerations
-Neonatalinfectionmayoccurviaverticaltransmissionfrommothertochild;
maternal history should be considered.
- Differential diagnosis is required from bronchitis caused by respiratory syn-
cytial virus (RSV), parainfluenza virus, human metapneumovirus, etc.
- Confirmation requires PCR testing or pathogen culture; treatment priori-
tizes macrolide antibiotics (e.g., erythromycin or azithromycin).
Reasoning content
The user is inquiring about symptoms of childhood chlamydia infection. I
need to retrieve relevant information from the provided knowledge base.
First,knowledgeindex199indicatesthatinneonatespresentingwithafebrile,
paroxysmal cough and IgM antibody titer >1:8, Chlamydia trachomatis
infection should be considered, with treatment using erythromycin or sul-
famethoxazole. The key symptoms here are absence of fever and paroxysmal
cough.
Next, knowledge index 192 states that in adolescents and adults with acute
bronchitis, Mycoplasma or Chlamydia infections may cause persistent cough
lasting over two weeks, typically without fever. Although this pertains to
older children and adults, pediatric Chlamydia infections may also present
similar symptoms, such as persistent cough.
Additionally, knowledgeindex332addressesneonatalsepsis, primarilybacte-
rial but potentially involving chlamydial infections—particularly Chlamydia
trachomatis—in newborns, presenting with afebrile cough.
Synthesizing this information, key symptoms of childhood chlamydia infec-
28

tion include fever-free, paroxysmal coughing, potentially accompanied by
persistent coughing lasting over two weeks, especially in neonates. Labora-
tory indicators may also show elevated IgM antibodies. Differential diagnosis
from other pathogens like mycoplasma or viruses is essential.
Appendix B.2. Reasoning Trajectory Data Sample
Figure B.11: Antibiotic Usage Distribution
Appendix C.
Appendix C.1. SFT training dataset sample
For a single row:
Age & gender
Female, 51 years old
Chief complaint
Fever for 12 days
29

Present illness
On2024-02-23, thepatientdevelopedsorethroatwithoutapparentcause, fol-
lowed by afternoon fever with Tmax 39.6°C. She experienced one daily fever
peak accompanied by chills and occasional coughing with small amounts of
white, mucous-like sputum. She reported mild periodontal pain but denied
abdominal pain, diarrhea, frequent urination, or urgency. Self-administered
oral cefotiam + throat inflammation powder + traditional Chinese medicine,
withminimalsymptomrelief. DuetoexcessivesweatingaftertakingTylenol,
the patient took it only once; subsequently, fever resolved spontaneously each
day. Consulted local dental department, where periodontal inflammation
was suspected. Discontinued aforementioned medications and switched to
ciprofloxacin 2 qd. Fever persisted daily without resolution. 03-01 Visit to
Tsinghua University Hospital: Blood count: WBC 7.38×10*9/L, NEUT%
62.5%, EOS% 0.06%, HGB 121g/L, MCV 91.5fl, MCHC 331g/L, PLT402∗
109/L Inflammatory markers: hsCRP 131.00 mg/L, SAA 350 mg/L, In-
fluenza A/B antigen (–), Mycoplasma pneumoniae + Chlamydia pneumoniae
IgM (–). Chest CT: Three ground-glass nodules measuring approximately
2–4 mm in longest diameter were visible in the apical segment, posterior seg-
ment of the right upper lobe, and dorsal segment of the right lower lobe. No
increased pulmonary markings bilaterally; no other significant abnormalities
noted. Presented to our emergency department for further evaluation.
Past medical historyOn 2024-02-06, presented with sore throat and fever.
Self-testedpositiveforCOVID-19antigen. Administeredoralnirmatrelvir/ritonavir.
Temperature normalized after 3 days, with subsequent antigen test turning
negative. In 2004, proteinuria detected during a physical examination led
to a kidney biopsy confirming IgA nephropathy. Regular monitoring was
initiated. In October 2021, due to 24-hour urine protein exceeding 0.5g/day,
mycophenolate mofetil dispersible tablets were added at 5g bid→0.75g qd
orally. This medication has since been discontinued. Concurrently, elevated
uric acid levels were identified, prompting oral administration of febuxostat,
which has also been discontinued. In 2020, ultrasound detected a thyroid
nodule; fine-needle aspiration showed no tumor cells.Diagnostic recom-
mendationsThe patient presented with fever, accompanied by sore throat
and toothache. A purulent coating was visible on the posterior pharyngeal
wall, suggesting possible pharyngitis or periodontitis. Physical examination
revealed tenderness over the left maxillary sinus, making maxillary sinusi-
tis a consideration. Cefuroxime 2g qd was added for infection control. and
Metronidazole 1g tid to cover anaerobic bacteria. Fever recurred for 2-3 days
30

with concomitant cough and rhinorrhea. Peripheral blood culture yielded no
definitivepathogens. Symptomsgraduallyresolved, withfevergraduallysub-
siding to normal levels. Cefuroxime and Metronidazole were discontinued,
replaced with Levofloxacin tablets 0.5g qd. Temperature remained consis-
tently normal.
Appendix C.2. Unstructured evaluation dataset sample
For a single row:
Question
Patient, male, 65 years old, admitted for "recurrent reducible mass in right
inguinal region for 2 years," diagnosed with "right inguinal hernia," sched-
uled for "laparoscopic tension-free inguinal hernia repair." No history of drug
allergies or other underlying conditions. How should prophylactic antibiotics
be administered?
Raw knowledge
{‘chunk-id’: c124, ‘content’: ‘Clean surgeries (such as inguinal hernia repair)
typically do not require routine prophylactic medication. However, for hernia
repairsinvolvingsyntheticmeshimplants, prophylacticantibioticsarerecom-
mended. Recommended regimen: A single intravenous dose of cefazolin 1-2g
administered 30 minutes preoperatively. Alternatively, a single intravenous
dose of ampicillin-sulbactam 3g, clindamycin 900mg, or vancomycin 1g may
be administered. No postoperative boost is required.’, ‘page-no’: 124}
Answer
"Cefazolin 1-2g" or "Amoxicillin-Sulbactam 3g" or "Clindamycin 900mg" or
"Vancomycin 1g"
Appendix C.3. MedQA data sample
31

Appendix C.4. PUMCH Antimicrobial Sample
For a single row:
Query
Patient, female, 58 years old, has been hospitalized multiple times for "com-
plicatedurinarytractinfection."Currenturinecultureresults: ESBL-producing
Escherichia coli resistant to ceftriaxone and fluoroquinolones, with suscepti-
bility only to carbapenems and amikacin. How should the treatment regimen
be selected?
Therapy
For severe infections caused by ESBL-producing Enterobacteriaceae (e.g.,
pyelonephritis, bacteremia), guidelines recommend carbapenems as standard
therapy. Options: IV infusion of Ertapenem 1g q24h or Meropenem 1g q8h
or Imipenem-Cilastatin 500mg q6h.
Keywords
"Ertapenem," "Meropenem," "Imipenem-Cilastatin"
32

Appendix C.5. Agentic Reinforcement Learning vs. Reinforcement Learning
References
[1] T. Brown, et al., Language models are few-shot learners, OpenAI Blog
(2020).
[2] T. Hulsen, Explainable artificial intelligence (xai): Concepts and chal-
lenges in healthcare, AI 4 (2023) 652–666.
[3] Y. Li, Y. Li, M. Wei, et al., Innovation and challenges of artificial intel-
ligence technology in personalized healthcare, Sci Rep 14 (2024) 18994.
33

[4] L. Celi, J. Cellini, M.-L. Charpignon, et al., Sources of bias in artifi-
cial intelligence that perpetuate healthcare disparities—a global review,
PLoS Digit. Health 1 (2022) e0000022.
[5] OpenAI, Gpt-4 technical report, Tech. Rep. arXiv:2303.08774 (2023).
[6] OpenAI, Gpt-4o system card (May 2024).
[7] J. Bai, et al., Qwen2-vl: Enhancing vision-language model’s percep-
tion of the world at any resolution, arXiv preprint (arXiv:2409.12191)
(September 2024).
[8] DeepSeek-AI, Deepseek-v2: A strong, economical, and efficient mixture-
of-experts language model, Tech. Rep. arXiv:2405.04434 (May 2024).
[9] Anthropic, Claude 3 model card and system report, Tech. rep. (March
2024).
[10] C. Wang, J. Zhang, N. Lassi, X. Zhang, Privacy protection in using
artificial intelligence for healthcare: Chinese regulation in comparative
perspective, Healthcare 10 (2022) 1878.
[11] B. Murdoch, Privacy and artificial intelligence: challenges for protecting
health information in a new era, BMC Med Ethics 22 (2021) 122.
[12] H. Zhang, J. Chen, M. Wang, H. Li, X. Zhou, Privacy leakage in large
language models: A comprehensive survey, arXiv preprint (2023).
[13] E. Ullah, A. Parwani, M. Baig, et al., Challenges and barriers of using
large language models (llm) such as chatgpt for diagnostic medicine with
a focus on digital pathology–a recent scoping review, Diagn Pathol 19
(2024) 43.
[14] M. DJ, C. MP, P. E, et al., Artificial intelligence in low- and middle-
income countries: Innovating global health radiology, Radiology 297 (3)
(2020) 513–520.
[15] F. J, G. L, C. VM, M.-I. C, L. K, Federated learning in low-resource set-
tings: A chest imaging study in africa — challenges and lessons learned,
arXiv (2025).
34

[16] W. D, H. H, N. A, P. A, Deep learning and its application for health-
care delivery in low and middle income countries, Frontiers in Artificial
Intelligence 4 (2021) 553987.
[17] Y. Huang, K. Tang, M. Chen, et al., A comprehensive survey on evalu-
ating large language model applications in the medical industry, arXiv
preprint (2024).
[18] K. Singhal, S. Azizi, T. Tu, et al., Large language models encode clinical
knowledge, Nature 620 (2023) 172–180.
[19] Z. Guo, R. Jin, C. Liu, et al., Evaluating large language models: A
comprehensive survey, arXiv preprint (2023).
[20] S. Yao, J. Zhao, D. Yu, et al., React: Synergizing reasoning and acting
in language models, in: International Conference on Learning Represen-
tations (ICLR), 2023.
[21] D. Guo, D. Yang, H. Zhang, et al., Deepseek-r1 incentivizes reasoning
in llms through reinforcement learning, Nature 645 (2025) 633–638.
[22] Z. Shao, P. Wang, Q. Zhu, et al., Deepseekmath: Pushing the limits of
mathematical reasoning in open language models (2024).
[23] E. Hu, Y. Shen, P. Wallis, et al., Lora: Low-rank adaptation of large
language models (2021).
[24] S. Rajbhandari, J. Rasley, O. Ruwase, Y. He, Zero: Memory optimiza-
tions toward training trillion parameter models (2020).
[25] J. Ren, S. Rajbhandari, R. Aminabadi, et al., Zero-offload: Democra-
tizing billion-scale model training (2021).
[26] B. M., K. R., M. H., et al., Communication-efficient jaccard similarity
for high-performance distributed genome comparisons, in: 2020 IEEE
International Parallel and Distributed Processing Symposium (IPDPS),
2020, pp. 1122–1132.
[27] B. Biswas, R. Ramnath, Efficient and interpretable information retrieval
for product question answering with heterogeneous data (2024).
35

[28] O. Khattab, M. Zaharia, Colbert: Efficient and effective passage search
via contextualized late interaction over bert (2020).
[29] C. J, X. S, Z. P, et al., Bge m3-embedding: Multi-lingual, multi-
functionality, multi-granularity text embeddings through self-knowledge
distillation (2023).
[30] Q. Yang, R. Wang, J. Chen, R. Su, T. Tan, Fine-tuning medical lan-
guage models for enhanced long-contextual understanding and domain
expertise (2024).
[31] F. Dorfner, A. Dada, F. Busch, et al., Biomedical large language models
seem not to be superior to generalist models on unseen medical data
(2024).
[32] R. Tinn, H. Cheng, Y. Gu, et al., Fine-tuning large neural language
models for biomedical natural language processing, Patterns (New York,
N.Y.) 4 (4) (2023) 100729.
36