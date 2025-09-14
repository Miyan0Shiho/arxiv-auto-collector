# PL-CA: A Parametric Legal Case Augmentation Framework

**Authors**: Ao Chang, Yubo Chen, Jun Zhao

**Published**: 2025-09-08 06:08:06

**PDF URL**: [http://arxiv.org/pdf/2509.06356v1](http://arxiv.org/pdf/2509.06356v1)

## Abstract
Conventional RAG is considered one of the most effective methods for
addressing model knowledge insufficiency and hallucination, particularly in the
judicial domain that requires high levels of knowledge rigor, logical
consistency, and content integrity. However, the conventional RAG method only
injects retrieved documents directly into the model's context, which severely
constrains models due to their limited context windows and introduces
additional computational overhead through excessively long contexts, thereby
disrupting models' attention and degrading performance on downstream tasks.
Moreover, many existing benchmarks lack expert annotation and focus solely on
individual downstream tasks while real-world legal scenarios consist of
multiple mixed legal tasks, indicating conventional benchmarks' inadequacy for
reflecting models' true capabilities. To address these limitations, we propose
PL-CA, which introduces a parametric RAG (P-RAG) framework to perform data
augmentation on corpus knowledge and encode this legal knowledge into
parametric vectors, and then integrates this parametric knowledge into the
LLM's feed-forward networks (FFN) via LoRA, thereby alleviating models' context
pressure. Additionally, we also construct a multi-task legal dataset comprising
more than 2000 training and test instances, which are all expert-annotated and
manually verified. We conduct our experiments on our dataset, and the
experimental results demonstrate that our method reduces the overhead
associated with excessively long contexts while maintaining competitive
performance on downstream tasks compared to conventional RAG. Our code and
dataset are provided in the appendix.

## Full Text


<!-- PDF content starts -->

PL-CA: A Parametric Legal Case Augmentation Framework
Ao Chang1,2, Yubo Chen1,2*, Jun Zhao1,2
1The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese
Academy of Sciences, Beijing, China
2School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China
Abstract
Conventional RAG is considered one of the most effective
methods for addressing model knowledge insufficiency and
hallucination, particularly in the judicial domain that requires
high levels of knowledge rigor, logical consistency, and con-
tent integrity. However, the conventional RAG method only
injects retrieved documents directly into the model’s context,
which severely constrains models due to their limited con-
text windows and introduces additional computational over-
head through excessively long contexts, thereby disrupting
models’ attention and degrading performance on downstream
tasks. Moreover, many existing benchmarks lack expert an-
notation and focus solely on individual downstream tasks
while real-world legal scenarios consist of multiple mixed
legal tasks, indicating conventional benchmarks’ inadequacy
for reflecting models’ true capabilities. To address these lim-
itations, we propose PL-CA, which introduces a parametric
RAG (P-RAG) framework to perform data augmentation on
corpus knowledge and encode this legal knowledge into para-
metric vectors, and then integrates this parametric knowl-
edge into the LLM’s feed-forward networks (FFN) via LoRA,
thereby alleviating models’ context pressure. Additionally,
we also construct a multi-task legal dataset comprising more
than 2000 training and test instances, which are all expert-
annotated and manually verified. We conduct our experiments
on our dataset, and the experimental results demonstrate that
our method reduces the overhead associated with excessively
long contexts while maintaining competitive performance on
downstream tasks compared to conventional RAG. Our code
and dataset are provided in the appendix.
Introduction
In recent years, the rapid development of LLM in the field
of natural language processing have led to their increas-
ingly widespread applications in legal AI scenarios (Mul-
lick et al. 2022; Kim, Jung, and Koo 2024; Fan et al. 2025).
The legal tasks, including Legal Judgment Prediction (LJP),
Statute Article Generation (SAR), Legal Document Genera-
tion (LDG), present greater challenges for the application of
LLMs. Consequently, LLMs must accurately generate legal
articles (Su et al. 2024), understand complex legal cases (Li
et al. 2024d), and make reasonable and accurate judgments
(He et al. 2024) across various legal tasks.
*Corresponding author, yubo.chen@nlpr.ia.ac.cn
Copyright © 2026, Association for the Advancement of Artificial
Fact:Reason:Articles:Judgment:The defendant, xxx was driving a non-valid small passenger car along…The court held that the defendant, xxx, violated the regulations on road traffic safety …The defendant, xxx, was found guilty of dangerous driving and sentenced to one month of detention and …Article 133-1, Paragraph 1, Item (2) of the Criminal Law of the People‘s Republic of China …Test Case
Fact:Reason:Articles:Judgment:xxx discovered during an inspection that nine pigs without quarantine identification and quarantine …The xxx Bureau detained the involved pigs solely on the grounds that they lacked livestock identification, and without ……xx’spigs and slaughtering them was illegal …Animal Epidemic Prevention Law of the People's Republic of China" Articles 29, 76, 97Focus:The dispute focus of this case is: whether the administrative act of sale was illegal…Parametric CaseFigure 1: The detailed structure of Test Cases and Parametric
Cases, which is used for parametric injection.
RAG has emerged as an effective strategy to mitigate
the limitations of LLMs concerning knowledge insufficiency
and hallucination issues by injecting relevant information
from external knowledge bases into the model context (Li,
Yuan, and Zhang 2024). Traditional RAG provides addi-
tional knowledge support to the model by directly incorpo-
rating retrieved documents into its context. To further im-
prove LLM’s performance, some works (Li et al. 2024b,
2023a; Xiao et al. 2021) focus on optimizing retrievers to
recall more gold documents. However, despite its success
in many domains, the limitations of RAG are becoming in-
creasingly evident within the legal field. For example, legal
texts often have exceptionally long contexts, and a single
legal case may contain thousands of tokens on average (Li
et al. 2023b). Conventional RAG methods, which directly
concatenate retrieved documents into the LLM’s input con-
text, present a significant bottleneck for models with lim-
Intelligence (www.aaai.org). All rights reserved.arXiv:2509.06356v1  [cs.CL]  8 Sep 2025

Dataset SimuCourt RareCases Legal-CA
cases 420 136 591
avg articles 3.71 5.90 6.71
avg length per case fact 346.1 491.3 605.8
avg token length 3326 5807 415
fine-tune? No Yes Yes
Table 1: Comparison with other datasets
ited context windows. Too long contexts not only introduce
substantial time and space overhead, thereby significantly
increasing inference costs, but also negatively influence the
model’s attention to crucial information, consequently de-
grading performance on downstream tasks (Li et al. 2024e;
Qian et al. 2024). More importantly, existing research in-
dicates that LLMs generally perform better when utilizing
its internal parametric knowledge (Yu and Ananiadou 2024)
than externally injected contextual knowledge.
Besides, current benchmarks (Li et al. 2025; Fan et al.
2025; Su et al. 2024; Sun et al. 2024; Chen et al. 2025b)
in the legal domain often lack high-quality expert annota-
tions and tend to focus solely on individual downstream
legal tasks. However, real-world legal scenarios typically
involve multiple interrelated tasks, making conventional
benchmarks inadequate to fully reflect the true capabilities
of LLMs in complex legal environments, thus failing to meet
the demands of practical legal applications.
Inspired by P-RAG (Su et al. 2025a), Parametric Retriever
Augmentation Generation, we propose PL-CA: aParametric
LegalCaseAugmentation framework to overcome the lim-
itations of traditional RAG in the legal domain and address
the deficiencies of current benchmarks. PL-CA introduces a
parametric RAG approach that augments legal corpora with
enhanced knowledge. This knowledge is then efficiently in-
tegrated into the LLM using Low-Rank Adaptation (LoRA)
techniques. This approach effectively reduces the burden on
the model’s context window, enabling it to internally encode
more legal knowledge and thereby improving its ability to
utilize such knowledge.
Furthermore, to enable a more comprehensive and accu-
rate evaluation of LLM performance in the legal domain, we
construct Legal-CA, a multi-task legal dataset. This dataset
comprises 590 expert-annotated and manually verified test
samples and 1,990 training samples, covering three legal
areas, criminal law, administrative law, and civil law, and
encompasses various downstream legal tasks, which is de-
signed to better reflect model performance in realistic, com-
plex legal scenarios.
Experimental results show that our PL-CA method re-
duces the computational cost caused by excessively long
contexts without compromising performance on down-
stream tasks compared to conventional RAG approaches.
Particularly in tasks requiring deep understanding and rea-
soning of legal knowledge—such as legal article retrieval
and judgment prediction—PL-CA shows significant advan-
tages. These findings strongly validate the effectiveness and
feasibility of P-RAG in the legal domain, offering new in-
sights and methodologies for the application of LLMs in le-Dataset civil crime admin
number of cases 17.55M 2.45M 0.87M
Table 2: Statistic of Legal-KD
gal practice.
Our contributions are summarized as follows:
•We have proposed the PL-CA framework, which is the
first to apply P-RAG to legal downstream tasks, allevi-
ate the issues in some degree caused by excessively long
legal texts in past tasks.
•We construct Legal-CA, a legal dataset comprising 590
test instances and 1990 train instances, along with Legal-
KD, a legal knowledge case corpus, to assess LLMs’ le-
gal capacity comprehensively.
•We conduct extensive experiments to validate the ef-
fectiveness of our methods. Our approach enables the
LLM to achieve significant improvements on down-
stream tasks, even surpassing GPT-4o in certain domains.
Related Work
Many works have emerged to assess models’ capacity to
handle legal problems. To assess models’ LJP ability, sev-
eral datasets like CAIL(Xiao et al. 2018) have been pro-
posed. Subsequently, with the advent of LLMs, several legal
benchmark datasets have been subsequently proposed to as-
sess LLMs’ capabilities in various downstream legal tasks,
including LegalBench(Pipitone and Alami 2024) from the
United States, LawBench(Fei et al. 2023) from China, and
LexFile(Chalkidis et al. 2023) from Europe.
Consequently, researchers begin to explore LLMs’ legal
capabilities in alternative scenarios. In the LJP task, (Wu
et al. 2023) enhanced LLM performance on downstream
tasks by fine-tuning models with legal precedents. In the Le-
gal Case Retrieval (LCR) domain, LeCard(Li et al. 2023b)
is proposed as a LCR dataset to evaluate retrieval models’
capacity. GEAR(Qin et al. 2024) integrates LJP and LCR
tasks by constructing document trees to improve recall capa-
bilities for legal cases. Furthermore, (Gao et al. 2024; Zhang
et al. 2025) optimize downstream task performance through
enhanced retrieval mechanisms, while (Sun et al. 2024) and
(Barron et al. 2025) adopt knowledge graph approaches to
enhance legal corpus retrieval capabilities. For the Statute
Article Retrieval (SAR) task, STARD(Su et al. 2024) con-
structed a comprehensive statute knowledge base, including
datasets and corpora. In contrast, SLARD(Su et al. 2024)
focuses on municipal regulations rather than statutory ar-
ticles and proposes a new benchmark suite. Across these
retrieval-oriented downstream tasks, model performance re-
mains generally suboptimal, revealing the gap between mod-
els’ legal knowledge comprehension and application capa-
bilities.
In the realm of agent-based legal applications, LLMs
have also demonstrated significant contributions. Agent-
Court(Chen et al. 2025a) enhances legal knowledge by uti-
lizing debate information generated by lawyer agents as

Large CorpusTime ConsumingTradition RAG Issues 
Hard forOnlineSearch
Tong Long ContextLose Attention
KnowledgeAugmentParametric RAG SolutionEncodingParametric injectionShort ContextBetter Capacity
Learning
Figure 2: Comparison between traditional RAG and P-RAG.
a corpus, enabling models to retrieve it when addressing
downstream tasks. SimuCourt(He et al. 2024) proposes new
datasets and extensive case data based on the debate frame-
work. ASP2LJ(Chang et al. 2025) improves agent debat-
ing capabilities by generating synthetic case data, mitigat-
ing limitations arising from insufficient existing legal cases
and long-tail distribution issues. MASER(Yue et al. 2025)
presents a more realistic scenario by not only simulating
multi-turn question-answering interactions in legal consul-
tation but also imparting clients with diverse personalities
based on the Big-5, thereby generating more realistic and
complex data.
Furthermore, beyond conventional tasks such as LJP and
LCR, researchers are expanding into novel legal scenarios
to comprehensively evaluate LLMs’ legal capabilities. (Li
et al. 2025) proposes the CaseGen benchmark to test mod-
els’ legal document generation abilities, while JuDGE(Su
et al. 2025b) integrates LJP with document generation to
analyze model performance across multiple dimensions. To
better approximate the complexity of real-world cases, (Li
et al. 2024c) introduces LegalAgentBench, which analyzes
LLM capabilities across 17 realistic legal scenarios and 37
external knowledge base interaction tools. In legal consulta-
tion tasks, LexRAG proposes a multi-turn legal consultation
framework and organizes it into an RAG toolkit, empower-
ing subsequent legal RAG research.
Methodology
Preliminary
As illustrated in Figure 1, a complete legal judgment docu-
ment typically comprises four main components: case facts,
legal reasoning, judgment results, and relevant statutes. Cor-
respondingly, typical legal judgment scenarios involve three
primary downstream tasks: LJP, SAG, and LDG. These tasks
are sequentially organized to produce legal responses based
on the provided factual information.
To evaluate the performance of LLMs on these tasks, we
define them as follows: given a case factf, an LLMΘis ex-
pected to generate a responser= Θ(f). In general, relyingModel(%) Recall@5 Recall@20 Recall@100
BM25 3.60 6.57 11.62
LawFormer 0.08 0.26 1.20
bge-m3 1.39 3.58 9.04
ChatLaw-Text2Vec 2.24 6.44 10.16
SAILER zh 0.18 0.39 1.24
Table 3: Different retrievers’ article recall in Legal-CA,
solely on the internal legal knowledge and reasoning capa-
bilities of LLMs often fails to produce satisfactory results.
Therefore, external knowledge is typically incorporated via
RAG to enhance performance.
Vanilla RAG.For legal retrieval tasks, this approach uses
the basic case factfas the input query. A sparse or dense re-
triever performs similarity matching betweenfand a corpus
D={d 1, d2, . . . , d n}, selecting the top-kdocuments as the
retrieved setD={d′
1, d′
2, . . . , d′
k}. The queryfand the re-
trieved documentsDare then concatenated and provided as
contextual input to the LLM. The model subsequently gen-
erates the final output asr= Θ(f, D).
While this approach is conceptually simple and easy to
implement, it often leads to performance degradation when
handling lengthy legal documents, due to the input length
limitations and context dilution in LLMs.
P-RAG.Su et al. (2025a) proposes a novel RAG
paradigm, P-RAG, which enhances LLMs’ ability to com-
prehend and utilize knowledge by injecting external knowl-
edge directly into model parameters. As illustrated in Fig-
ure 2, we employ P-RAG to improve LLMs’ performance,
which is better than traditional RAG.
∆Θ =Encode 
{di}k
i=1
,
Θ′= Θ + ∆Θ.(1)
As shown in Equation 1, the retrieved documents are
encoded and used to update the parameters of the LLM,
thereby internalizing external knowledge. Recent stud-
ies (Yu and Ananiadou 2024) have demonstrated that
LLMs are generally more effective at leveraging internalized
knowledge than contextually provided external information.
In the legal domain, where accurate and efficient legal rea-
soning is critical, this characteristic is particularly valuable.
To improve LLM performance in legal tasks, we therefore
propose to augment and parameterize legal knowledge. Fol-
lowing the design in Su et al. (2025a), we divide the proce-
dure into two stages: offline and online.
However, given the large scale of the legal corpus, fully
parameterizing the entire dataset would introduce significant
computational overhead. Thus, in the offline stage, we select
1,990 representative legal samples from the authoritative le-
gal website PKULaw to construct the offline corpusD offline.
For the online stage, we introduce a new corpus, Legal-KD,
as the online knowledge baseD online.

gpt-4o-miniRewriteFactFocusReasonArticlesJudgment
……
……
…………
……encodeFact ⊕ focus
Fact ⊕ reasonFact ⊕ judgeFact ⊕ articlesInjection
Data AugmentParametric
Permute
Figure 3: Overview of the Case Augmentation and LLM Injection Pipeline. Our method involves five document sections:
fact, focus, reason, judgment, and article. Each section is rewritten by GPT-4o-mini for case augmentation. The fact section is
then concatenated with the other augmented sections, and the combined item is encoded into embeddings. Finally, LoRA is
employed to inject these embeddings into the LLM.
Offline P-RAG
Given an LLMΘ, an offline corpusD offline, and a set of
queriesQ={q 1, q2, . . . , q n}, where each queryq iis paired
with a corresponding answera i, the dataset can be repre-
sented as(Q,A) ={(q i, ai)}.
The structure of each legal case in the offline corpus is
illustrated in Figure 1. To improve the model’s generaliza-
tion capability with respect to legal knowledge, we perform
data augmentation on each component of the case structure,
including: case facts, legal reasoning, dispute focus, appli-
cable statutes, and judgment outcome. In this setup, the case
fact is treated as the query, while the remaining components
serve as corresponding answers.
For the data augmentation process, we employ GPT-4o-
mini to generate diverse variants of each data point. Each
sample is rewritten three times, ensuring that the core le-
gal information remains unchanged—this includes charges,
sentencing outcomes, fines, and other legally significant de-
tails (e.g., names, locations, and terminology). As a result,
the size of each component increases by a factor of four, and
the number of QA pairs per case is expanded by a factor
of sixteen (4 components × 4 total variations), thus signifi-
cantly enhancing the model’s ability to internalize and rea-
son about legal knowledge. The augmentation process for
each component is detailed as follows:
•Case Fact: Preserve charges, sentencing, and key factual
details (e.g., monetary amounts, locations), while rewrit-
ing the text using alternative legal terminology.
•Reason: Rephrase sentence structures without altering
the underlying legal logic (e.g., rewriting “constitutes
theft” as “meets the constitutive requirements of theft”).
•Focus: Substitute key dispute phrases with synonymous
legal expressions (e.g., “whether it constitutes voluntary
surrender” as “whether it satisfies the condition of volun-
tary surrender”).•Article: Retain statute numbers while paraphrasing the
legal provisions using varied expressions.
•Judgment: Keep the judgment outcome consistent, but
vary the phrasing (e.g., “sentenced to three years’ im-
prisonment” as “sentenced to imprisonment for a term of
three years”).
After obtaining the augmented datasetD′={(q′, a′)},
we proceed with parameterized knowledge injection into the
LLM. Specifically, we adopt Low-Rank Adaptation (LoRA)
and tailor it to the legal domain to construct the PL-CA
pipeline. The procedure consists of the following steps:
•LoRA Parameter Initialization: For each legal case
di∈ D′, we initialize a pair of low-rank matricesA i∈
Rh×randB i∈Rk×r, wherehis the hidden dimen-
sion,kis the feed-forward network (FFN) intermediate
dimension, andr≪min(h, k)denotes the LoRA rank.
•Training Objective: For each QA pair(q′, a′)∈ D′,
we concatenate the question and answer into an input se-
quencex= [q′⊕a′], where⊕denotes token-level con-
catenation. We use the standard next-token prediction ob-
jective to optimize the trainable LoRA parameters. LetΘ
denote the frozen parameters of the pre-trained LLM, and
∆Θi={A i, Bi}represent the trainable LoRA adapters.
The training objective is defined as:
min
∆Θ iX
(q′,a′)∈D′
iTX
t=1−logP Θ+∆Θ i(xt|x<t)(2)
•Parameter Merging: After training, each legal cased i
is associated with a specific set of LoRA parameters
∆Θi={A i, Bi}. These parameters are stored as the
case-specific, parameterized representation of the legal
knowledge embedded ind i.

ModelLA-P LA-R LA-F1 Charge Imprison Probation Fine CA-P CA-R CA-F1
LexiLaw 15.82 9.92 12.19 80.91 37.95 38.58 37.25 3.26 3.66 3.45
ChatLaw 8.91 5.31 6.66 73.91 27.92 38.95 30.25 7.08 9.04 7.94
gpt-4o-mini 4.45 4.32 4.38 66.85 38.82 36.59 39.21 9.85 13.18 11.27
gpt-3.5-turbo-0125 4.19 3.39 3.75 78.42 34.75 37.53 37.17 14.22 23.29 17.66
gpt-4o-2024-11-20 17.28 13.94 15.43 88.62 43.42 40.79 43.6325.75 31.54 28.35
Qwen1.5-7B-Chat 15.3 10.78 12.65 78.92 37.62 38.45 37.61 4.45 7.36 5.54
Qwen1.5-7B-Chat+RAG 20.35 13.96 16.56 87.78 39.67 38.85 40.12 10.48 15.46 12.49
PL-CA 33.14 18.83 24.01 92.3943.40 40.52 41.53 27.5327.06 27.29
Combine Both48.2514.41 22.19 87.22 40.57 38.73 40.38 16.63 18.49 17.51
Table 4: Fine-grained performance of Legal-CA. The prefix ”LA” means legal article, ”CA” means Civil & Admin.
ModelJ-P J-R J-F1 R-P R-R R-F1
LexiLaw 62.96 65.68 64.10 44.29 72.37 54.95
ChatLaw 64.63 73.03 68.34 39.53 74.51 50.74
gpt-4o-mini 61.69 78.06 68.67 57.43 79.46 66.24
gpt-3.5-turbo-0125 65.02 75.32 69.61 47.22 78.51 57.44
gpt-4o-2024-11-20 55.06 81.49 65.1176.99 76.01 76.75
Qwen1.5-7B-Chat 63.94 79.42 70.67 53.72 74.15 63.26
Qwen1.5-7B-Chat+RAG 48.86 80.10 57.34 53.43 77.86 62.91
PL-CA(P-RAG)76.0172.5273.9472.76 75.48 73.52
Combine Both 37.9182.4850.14 72.27 75.03 72.83
Table 5: Coarse-grained performance of Legal-CA. The prefix ”J” means judgment, and ”R” means reason.
Online P-RAG
After completing the offline preprocessing of the corpus,
the model acquires a foundational level of legal knowledge.
Similar to the offline PRAG method, the online stage also
performs data augmentation, with the key distinction being
the incorporation of a retrieval component.
Retrieval: The retriever performs online retrieval over the
online corpus, selecting the top-1 most relevant case and ex-
tracting the corresponding legal statutes from the top-5 re-
trieved cases. These are used as the basis for subsequent on-
line parameterized injection.
Parametric Injection: Following the parameterization
strategy from the offline stage, the retrieved case and legal
articles are structurally reformatted. An example of the on-
line case structure is illustrated in Figure 1. It is worth noting
that, due to structural differences between the offline and on-
line corpora, the offline corpus is decomposed into four com-
ponents: fact, reasoning, judgment, and applicable statutes.
After applying data augmentation, the parameterized update
is carried out according to Equation 2, and the resulting pa-
rameters are injected into the LLM’s internal representation.
Θfinal= Θ′+ ∆Θ′(3)
Generation Stage: After completing both offline and on-
line parameter injections, the final model parametersΘ final
integrate legal knowledge from the entire corpus. The model
is now ready to be directly applied to downstream tasks for
experimental evaluation.Experiment Setting
Collection and Annotation
We constructLegal-CA, a multi-task benchmark compris-
ing 1,990 training instances and 590 test instances. Detailed
statistics are presented in Table 8. To support legal knowl-
edge retrieval, we also introduce a corpus namedLegal-KD,
which contains all criminal, administrative, and civil cases
published between 2018 and 2021. All documents have been
rigorously anonymized to ensure compliance with ethical
and privacy standards. Additionally, three Master of Laws
students, each having passed the National Legal Professional
Qualification Examination, are invited to assist in the quality
evaluation of the collected cases.
Parametric SetTo ensure the structure, authority, and
professional quality of our dataset, we collected 4,000 legal
documents from PKULaw, each annotated by certified legal
professionals. These documents include structured elements
such as case descriptions, key issues, legal reasoning, judg-
ment outcomes, and case analyses, as illustrated in Figure 1.
After rigorous filtering based on quality, representativeness,
and legal complexity, a total of 1,990 high-quality cases are
retained for parametric injection.
Test SetWe collected 800 legal cases from the official
Wenshu Court, covering criminal, administrative, and civil
domains. To prevent data leakage, only documents published
after January 1, 2025, were selected. Furthermore, to ensure
sufficient case complexity, we required the factual descrip-
tions to exceed 200 Chinese characters. After filtering, 590

ModelLA-P LA-R LA-F1 Charge Imprison Probation Fine CA-P CA-R CA-F1
Qwen1.8B 6.82 7.13 6.97 66.67 38.80 35.36 36.39 3.26 7.85 4.87
Qwen1.8B + RAG 10.58 10.4910.54 81.82 40.32 38.0739.6614.87 17.80 16.21
Qwen1.8B + P-RAG 10.14 6.12 7.63 73.84 40.12 38.25 39.0422.4222.68 22.55
Combine Both13.929.0110.94 86.91 40.53 39.7739.09 21.33 23.5622.39
Table 6: Fine-grained performance of Qwen1.5-1.8B-Chat in Legal-CA framework.
judge_P judge_R judge_f1 reason_P reason_R reason_f150556065707580Values
RAG and Structure
RAG
Structure
judge_P judge_R judge_f1 reason_P reason_R reason_f14050607080Values
PRAG+RAG and PRAG+Structure
PRAG+RAG
PRAG+Structure
Figure 4: The performance of plain RAG and structure RAG.
cases were retained. As shown in Table 1, compared with
existing datasets, Legal-CA ranks among the top in terms of
average token count and number of relevant articles. Each
case is manually annotated by the three law students, who
segment the documents into the following components: ba-
sic case facts, legal reasoning, judgment outcomes, and rele-
vant articles, consistent with the structure shown in Figure 1.
Legal Knowledge BaseWe propose Legal-KD, a legal
case knowledge database constructed from the corpus of ju-
dicial documents published by the Wenshu Court, encom-
passing all Chinese legal cases between 2018 and 2021. This
corpus serves as the source for online retrieval in our le-
gal case retrieval framework. All documents in the dataset
are unstructured plain texts. Given the large volume of judg-
ments, we limit the number of documents per cause of ac-
tion to a maximum of 10, and exclude any cases containing
fewer than 150 Chinese characters to ensure data quality. As
shown in Table 3, the BM25 retrieval method consistently
achieves the best performance; therefore, we adopt BM25 to
retrieve relevant cases. Implementation details and further
processing procedures are available in our code.
Metric
LJPFor criminal law cases, we evaluate the model’s pre-
dictions based on charges, prison terms (including imprison-
ment and probation), and fines. For administrative and civil
cases, we focus on key outcomes such as the plaintiff’s lit-
igation result, monetary compensation, and ownership dis-
putes. However, accurately predicting prison terms or fines
remains challenging for LLMs. As noted in prior studies (He
et al. 2024; Chang et al. 2025), criminal case descriptions of-
ten include the prosecution’s sentencing recommendations,
allowing models to simply copy this information to achieve
high accuracy. This shortcut fails to reflect the model’s true
reasoning ability. To ensure a fair evaluation of generated
content, we remove the prosecution’s claims from the case
fact descriptions and introduce a new evaluation metric, de-fined as follows (whereddenotes the difference,rthe refer-
ence answer, andhthe hypothesis):
d(r, h) = 1−1
1 + exp
−|r−h|
|r|+|h|+ϵ (4)
SAGSimilar to SAR, SAG focus on automatically gener-
ating the most relevant legal articles based on case descrip-
tions and retrieved documents. Given an input query, the
LLM is required to generate relevant legal articles and de-
termine whether the generated provisions are encompassed
within the correct set of legal articles.
LDGThis task not only requires correct judgment out-
comes but also emphasizes the interpretability and legal
validity of the reasoning process. Models need to pro-
duce human-like documents based on basic case facts. To
assess the model’s legal language generation capabilities,
we focus on the semantic alignment between generated
and ground-truth legal documents. Specifically, we adopt
thechinese-roberta-wwm-extmodel (Cui et al.
2019) to evaluate semantic relevance, thereby reflecting the
model’s effectiveness in legal expression.
Baseline
We choose several general or legal models as our baselines:
•Vanilla: We adopt Qwen1.5-7B-Chat as the base model.
For comparison, we also include GPT-3.5-turbo-0125,
GPT-4o-mini, and GPT-4o-2024-11-20.
•LexiLaw(Li et al. 2024a): A model fine-tuned on large-
scale Chinese legal corpora, demonstrating strong le-
gal knowledge and reasoning capabilities. It is based on
ChatGLM-6B.
•ChatLaw(Cui et al. 2024): We adopt ChatLaw-13B,
built upon Ziya-LLaMA-13B-v1, which performs well
across various Chinese legal tasks.
Hyperparameters
For all models, the temperature is set to 0.7. For LoRA
fine-tuning, we set thelearning rateto1×10−5,
lora rankto 2,lora alphato 32, and the number of
training epochs to 1. Further implementation details are pro-
vided in our code.
Results and Analysis
Main Results
In this section, we conduct a comprehensive experiment and
present its results in Table 4 and 5. We categorize the models

Model(%) LA-F1 Charge Imprison Probation Fine CA-F1 J-F1 R-F1
ONLINE 29.49 88.19 43.67 39.45 41.81 31.23 76.47 74.7
OFFLINE 16.47 84.32 42.05 40.47 40.43 19.38 69.96 72.5
Table 7: Performance of online and offline. ”LA” means legal article, ”CA” means Civil & Admin, and ”R” means Reason.
Features Crime-1 Admin-1 Civil-1 Crime-2 Admin-2 Civil-2
cases 192 203 196 697 266 1027
average articles 6.7 7.0 6.5 2.6 2.8 3.4
max articles 15 23 21 18 12 20
total articles 1280 1415 1271 1821 745 3457
avg len per case fact 440.9 384.7 482.1 430.3 560.7 700.0
Table 8: Comparison with other datasets. The suffix ”-1”
means test set, and ”-2” means train set.
into four groups: base models, vanilla RAG, P-RAG, and the
paradigm combining vanilla with P-RAG. The evaluation is
conducted across both coarse-grained and fine-grained tasks.
Our key findings are as follows:
(1) P-RAG exhibits a positive effect in enhancing model
capabilities. Compared with GPT-4o, which is currently the
most powerful model, our PL-CA, built upon Qwen1.5-7B-
Chat, achieves overall superior performance. Notably, in the
SAG task, the F1 score of PL-CA exceeds that of GPT-4o by
approximately 8.5 points. On more challenging tasks, such
as predicting prison terms and amounts, PL-CA also out-
performs all baselines except GPT-4o. These results demon-
strate that parameter-level knowledge injection can signifi-
cantly improve model capabilities, offering valuable insights
for the design of future knowledge integration methods.
(2) To better compare vanilla RAG and P-RAG, we
present the results of the base model equipped with RAG in
the Table 4. P-RAG consistently outperforms vanilla RAG,
suggesting that LLMs are more effective at utilizing para-
metric knowledge than context-based information. This ob-
servation aligns with prior findings in (Yu and Ananiadou
2024).
Ablation
StructureAs shown in Table 5, we observe that Qwen’s
performance on coarse-grained metrics declined after in-
corporating retrieved contextual information, regardless of
whether P-RAG is applied. We hypothesize that this perfor-
mance degradation may stem from the unstructured nature
of the retrieved legal documents, which are stored as free-
text strings without explicit annotation or segmentation.
To investigate this hypothesis, we reprocessed the re-
trieved corpus into a structured format, explicitly annotating
key legal elements such as case facts, legal reasoning, judg-
ments, and relevant legal articles. These structured compo-
nents were then incorporated into the model’s input context.
The corresponding experimental results are presented in Fig-
ure 4.
Interestingly, we found that the performance improve-
ments achieved through structured retrieval documents were
comparable to those obtained using unstructured ones. This
finding suggests that simply injecting structured content intothe model’s context does not substantially enhance its se-
mantic understanding or representation capability. To more
effectively align the model’s output with human-like le-
gal reasoning, alternative strategies such as supervised fine-
tuning or reinforcement learning may be necessary. We leave
the exploration of these directions to future work.
Offline and Online
To better measure the effects of online and offline learn-
ing, we trained Qwen1.5-7B separately using only online
and offline. To obtain more objective experimental results,
we reduced the amount of offline learning data to 600 sam-
ples, making it approximately equal to the amount of online
learning data. The results are shown in the table. It is in-
dicated that both online learning and offline learning have
positive effects on the model. Due to the higher semantic
relevance between the query and the retrieved online cases,
online learning proves more effective than offline learning
under equal data scale conditions. Therefore, online learn-
ing results in greater performance improvements.
Parameter Scale
Furthermore, we selected the smaller model Qwen1.5-1.8B-
Chat to investigate whether P-RAG could enhance the ca-
pabilities of small models. As shown in Table 6, although
the 1.8B model has limited capabilities, P-RAG still leads
to improvements. Interestingly, we observe that while P-
RAG brings improvements to the 1.8B model, its perfor-
mance gains are less obvious compared to traditional RAG
and the combined approach, which contrasts with the find-
ings for the 7B model. This suggests that P-RAG is more
effective for larger-scale models, whereas smaller models
benefit more from direct context injection. This observation
calls for further qualitative and quantitative analysis.
Conclusion
In this work, we proposePL-CA, a novel parametric le-
gal case augmentation framework that integrates LLM with
parametric knowledge injection to mitigate the limitations of
conventional context-based retrieval in legal AI. To system-
atically evaluate the performance of LLMs across a broad
spectrum of legal tasks, we also constructLegal-CA, a
comprehensive benchmark comprising both coarse-grained
and fine-grained subtasks. Legal-CA reflects real-world le-
gal challenges such as sentencing prediction, dispute focus
identification, and structured judgment generation.
Extensive experiments demonstrate that our method,
based on Qwen1.5-7B-Chat, achieves superior overall per-
formance compared to traditional RAG methods, as well as
powerful closed-source models like GPT-4o. Our findings

highlight the effectiveness of parametric injection in enhanc-
ing the legal capabilities of LLMs, especially on tasks in-
volving complex legal semantics. This work offers valuable
insights for future research in developing scalable, high-
performance legal AI systems.
References
Barron, R. C.; Eren, M. E.; Serafimova, O. M.; Matuszek,
C.; and Alexandrov, B. S. 2025. Bridging Legal Knowl-
edge and AI: Retrieval-Augmented Generation with Vector
Stores, Knowledge Graphs, and Hierarchical Non-negative
Matrix Factorization. arXiv:2502.20364.
Chalkidis, I.; Garneau, N.; Goanta, C.; Katz, D. M.; and
Søgaard, A. 2023. LeXFiles and LegalLAMA: Facilitating
English Multinational Legal Language Model Development.
arXiv:2305.07507.
Chang, A.; Zhou, T.; Chen, Y .; Qiu, D.; Liu, S.; Liu,
K.; and Zhao, J. 2025. ASP2LJ : An Adversarial
Self-Play Laywer Augmented Legal Judgment Framework.
arXiv:2506.18768.
Chen, G.; Fan, L.; Gong, Z.; Xie, N.; Li, Z.; Liu, Z.; Li, C.;
Qu, Q.; Alinejad-Rokny, H.; Ni, S.; and Yang, M. 2025a.
AgentCourt: Simulating Court with Adversarial Evolvable
Lawyer Agents. arXiv:2408.08089.
Chen, Z.; Ren, P.; Sun, F.; Wang, X.; Li, Y .; Zhao, S.; and
Yang, T. 2025b. SLARD: A Chinese Superior Legal Arti-
cle Retrieval Dataset. In Rambow, O.; Wanner, L.; Apidi-
anaki, M.; Al-Khalifa, H.; Eugenio, B. D.; and Schockaert,
S., eds.,Proceedings of the 31st International Conference
on Computational Linguistics, 740–754. Abu Dhabi, UAE:
Association for Computational Linguistics.
Cui, J.; Ning, M.; Li, Z.; Chen, B.; Yan, Y .; Li, H.;
Ling, B.; Tian, Y .; and Yuan, L. 2024. Chatlaw: A
Multi-Agent Collaborative Legal Assistant with Knowl-
edge Graph Enhanced Mixture-of-Experts Large Language
Model. arXiv:2306.16092.
Cui, Y .; Che, W.; Liu, T.; Qin, B.; Yang, Z.; Wang, S.; and
Hu, G. 2019. Pre-Training with Whole Word Masking for
Chinese BERT.arXiv preprint arXiv:1906.08101.
Fan, W.; Zheng, T.; Hu, Y .; Deng, Z.; Wang, W.; Xu, B.;
Li, C.; Li, H.; Shen, W.; and Song, Y . 2025. Legal Rule
Induction: Towards Generalizable Principle Discovery from
Analogous Judicial Precedents. arXiv:2505.14104.
Fei, Z.; Shen, X.; Zhu, D.; Zhou, F.; Han, Z.; Zhang, S.;
Chen, K.; Shen, Z.; and Ge, J. 2023. LawBench: Bench-
marking Legal Knowledge of Large Language Models.
arXiv:2309.16289.
Gao, C.; Xiao, C.; Liu, Z.; Chen, H.; Liu, Z.; and
Sun, M. 2024. Enhancing Legal Case Retrieval via
Scaling High-quality Synthetic Query-Candidate Pairs.
arXiv:2410.06581.
He, Z.; Cao, P.; Wang, C.; Jin, Z.; Chen, Y .; Xu, J.; Li,
H.; Jiang, X.; Liu, K.; and Zhao, J. 2024. AgentsCourt:
Building Judicial Decision-Making Agents with Court
Debate Simulation and Legal Knowledge Augmentation.
arXiv:2403.02959.Kim, M.; Jung, H.; and Koo, M.-W. 2024. SELF-
EXPERTISE: Knowledge-based Instruction Dataset Aug-
mentation for a Legal Expert Language Model. In Duh, K.;
Gomez, H.; and Bethard, S., eds.,Findings of the Associ-
ation for Computational Linguistics: NAACL 2024, 1098–
1112. Mexico City, Mexico: Association for Computational
Linguistics.
Li, H.; Ai, Q.; Chen, J.; Dong, Q.; Wu, Y .; Liu, Y .;
Chen, C.; and Tian, Q. 2023a. SAILER: Structure-aware
Pre-trained Language Model for Legal Case Retrieval.
arXiv:2304.11370.
Li, H.; Ai, Q.; Dong, Q.; and Liu, Y . 2024a. Lexilaw: A
Scalable Legal Language Model for Comprehensive Legal
Understanding.
Li, H.; Ai, Q.; Han, X.; Chen, J.; Dong, Q.; Liu, Y .; Chen,
C.; and Tian, Q. 2024b. DELTA: Pre-train a Discrimina-
tive Encoder for Legal Case Retrieval via Structural Word
Alignment. arXiv:2403.18435.
Li, H.; Chen, J.; Yang, J.; Ai, Q.; Jia, W.; Liu, Y .; Lin, K.;
Wu, Y .; Yuan, G.; Hu, Y .; Wang, W.; Liu, Y .; and Huang, M.
2024c. LegalAgentBench: Evaluating LLM Agents in Legal
Domain. arXiv:2412.17259.
Li, H.; Chen, Y .; Ai, Q.; Wu, Y .; Zhang, R.; and Liu,
Y . 2024d. LexEval: A Comprehensive Chinese Le-
gal Benchmark for Evaluating Large Language Models.
arXiv:2409.20288.
Li, H.; Shao, Y .; Wu, Y .; Ai, Q.; Ma, Y .; and Liu, Y . 2023b.
LeCaRDv2: A Large-Scale Chinese Legal Case Retrieval
Dataset. arXiv:2310.17609.
Li, H.; Ye, J.; Hu, Y .; Chen, J.; Ai, Q.; Wu, Y .; Chen, J.;
Chen, Y .; Luo, C.; Zhou, Q.; and Liu, Y . 2025. CaseGen: A
Benchmark for Multi-Stage Legal Case Documents Genera-
tion. arXiv:2502.17943.
Li, J.; Yuan, Y .; and Zhang, Z. 2024. Enhancing LLM Fac-
tual Accuracy with RAG to Counter Hallucinations: A Case
Study on Domain-Specific Queries in Private Knowledge-
Bases. arXiv:2403.10446.
Li, T.; Zhang, G.; Do, Q. D.; Yue, X.; and Chen, W. 2024e.
Long-context LLMs Struggle with Long In-context Learn-
ing. arXiv:2404.02060.
Mullick, A.; Nandy, A.; Kapadnis, M. N.; Patnaik, S.;
Raghav, R.; and Kar, R. 2022. An evaluation frame-
work for legal document summarization.arXiv preprint
arXiv:2205.08478.
Pipitone, N.; and Alami, G. H. 2024. LegalBench-RAG:
A Benchmark for Retrieval-Augmented Generation in the
Legal Domain. arXiv:2408.10343.
Qian, H.; Liu, Z.; Zhang, P.; Mao, K.; Zhou, Y .; Chen, X.;
and Dou, Z. 2024. Are Long-LLMs A Necessity For Long-
Context Tasks? arXiv:2405.15318.
Qin, W.; Cao, Z.; Yu, W.; Si, Z.; Chen, S.; and Xu, J. 2024.
Explicitly Integrating Judgment Prediction with Legal Doc-
ument Retrieval: A Law-Guided Generative Approach. In
Proceedings of the 47th International ACM SIGIR Con-
ference on Research and Development in Information Re-
trieval, SIGIR 2024, 2210–2220. ACM.

Su, W.; Hu, Y .; Xie, A.; Ai, Q.; Bing, Q.; Zheng, N.; Liu,
Y .; Shen, W.; and Liu, Y . 2024. STARD: A Chinese Statute
Retrieval Dataset Derived from Real-life Queries by Non-
professionals. In Al-Onaizan, Y .; Bansal, M.; and Chen,
Y .-N., eds.,Findings of the Association for Computational
Linguistics: EMNLP 2024, 10658–10671. Miami, Florida,
USA: Association for Computational Linguistics.
Su, W.; Tang, Y .; Ai, Q.; Yan, J.; Wang, C.; Wang, H.; Ye,
Z.; Zhou, Y .; and Liu, Y . 2025a. Parametric Retrieval Aug-
mented Generation. arXiv:2501.15915.
Su, W.; Yue, B.; Ai, Q.; Hu, Y .; Li, J.; Wang, C.; Zhang,
K.; Wu, Y .; and Liu, Y . 2025b. JuDGE: Benchmarking
Judgment Document Generation for Chinese Legal System.
arXiv:2503.14258.
Sun, J.; Dai, C.; Luo, Z.; Chang, Y .; and Li, Y .
2024. LawLuo: A Multi-Agent Collaborative Frame-
work for Multi-Round Chinese Legal Consultation.
arXiv:2407.16252.
Wu, Y .; Zhou, S.; Liu, Y .; Lu, W.; Liu, X.; Zhang, Y .; Sun,
C.; Wu, F.; and Kuang, K. 2023. Precedent-Enhanced Legal
Judgment Prediction with LLM and Domain-Model Collab-
oration. arXiv:2310.09241.
Xiao, C.; Hu, X.; Liu, Z.; Tu, C.; and Sun, M. 2021. Law-
former: A Pre-trained Language Model for Chinese Legal
Long Documents. arXiv:2105.03887.
Xiao, C.; Zhong, H.; Guo, Z.; Tu, C.; Liu, Z.; Sun, M.;
Feng, Y .; Han, X.; Hu, Z.; Wang, H.; and Xu, J. 2018.
CAIL2018: A Large-Scale Legal Dataset for Judgment Pre-
diction. arXiv:1807.02478.
Yu, Z.; and Ananiadou, S. 2024. Neuron-Level Knowledge
Attribution in Large Language Models. arXiv:2312.12141.
Yue, S.; Huang, T.; Jia, Z.; Wang, S.; Liu, S.; Song, Y .;
Huang, X.; and Wei, Z. 2025. Multi-Agent Simulator
Drives Language Models for Legal Intensive Interaction.
arXiv:2502.06882.
Zhang, K.; Yu, W.; Dai, S.; and Xu, J. 2025. Cita-
Law: Enhancing LLM with Citations in Legal Domain.
arXiv:2412.14556.