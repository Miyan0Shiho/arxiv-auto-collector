# LARGE: Legal Retrieval Augmented Generation Evaluation Tool

**Authors**: Minhu Park, Hongseok Oh, Eunkyung Choi, Wonseok Hwang

**Published**: 2025-04-02 15:45:03

**PDF URL**: [http://arxiv.org/pdf/2504.01840v1](http://arxiv.org/pdf/2504.01840v1)

## Abstract
Recently, building retrieval-augmented generation (RAG) systems to enhance
the capability of large language models (LLMs) has become a common practice.
Especially in the legal domain, previous judicial decisions play a significant
role under the doctrine of stare decisis which emphasizes the importance of
making decisions based on (retrieved) prior documents. However, the overall
performance of RAG system depends on many components: (1) retrieval corpora,
(2) retrieval algorithms, (3) rerankers, (4) LLM backbones, and (5) evaluation
metrics. Here we propose LRAGE, an open-source tool for holistic evaluation of
RAG systems focusing on the legal domain. LRAGE provides GUI and CLI interfaces
to facilitate seamless experiments and investigate how changes in the
aforementioned five components affect the overall accuracy. We validated LRAGE
using multilingual legal benches including Korean (KBL), English (LegalBench),
and Chinese (LawBench) by demonstrating how the overall accuracy changes when
varying the five components mentioned above. The source code is available at
https://github.com/hoorangyee/LRAGE.

## Full Text


<!-- PDF content starts -->

LRAGE: Legal Retrieval Augmented Generation
Evaluation Tool
Minhu Park1,*Hongseok Oh1,*Eunkyung Choi1Wonseok Hwang1,2,†
1University of Seoul2LBOX
{alsgn2003, cxv0519, rmarud202, wonseok.hwang}@uos.ac.kr
Abstract
Recently, building retrieval-augmented gener-
ation (RAG) systems to enhance the capabil-
ity of large language models (LLMs) has be-
come a common practice. Especially in the
legal domain, previous judicial decisions play
a significant role under the doctrine of stare
decisis which emphasizes the importance of
making decisions based on (retrieved) prior
documents. However, the overall performance
of RAG system depends on many components:
(1) retrieval corpora, (2) retrieval algorithms,
(3) rerankers, (4) LLM backbones, and (5)
evaluation metrics. Here we propose LRAGE ,
an open-source tool for holistic evaluation of
RAG systems focusing on the legal domain.
LRAGE provides GUI and CLI interfaces to
facilitate seamless experiments and investigate
how changes in the aforementioned five compo-
nents affect the overall accuracy. We validated
LRAGE using multilingual legal benches in-
cluding Korean (KBL), English (LegalBench),
and Chinese (LawBench) by demonstrating
how the overall accuracy changes when vary-
ing the five components mentioned above. The
source code is available at https://github.
com/hoorangyee/LRAGE .
1 Introduction
Recently large language models (LLMs) have
demonstrated remarkable performance across a
wide range of tasks. However, in expert domains–
where average users struggle to assess the accuracy
of an LLMs’ responses–their performance remains
limited due to a tendency to hallucinate (Dahl et al.,
2024; Magesh et al., 2024).
To address this limitation, it has become stan-
dard practice to employ Retrieval-Augmented Gen-
eration (RAG), which integrates LLMs with infor-
mation retrieval techniques. Although RAG sys-
tems have proven effective, they still exhibit hal-
*Equal contribution.
†Corresponding author.
Figure 1: Comparison of conventional RAG evalua-
tion pipeline (bottom) and the LRAGE framework (up)
where each process is seamlessly integrated.
lucinations (Magesh et al., 2024; Niu et al., 2024).
This underscores the need for rigorous evaluation
before introducing such systems to users, espe-
cially within expert domains.
Evaluating LLMs on domain-specific bench-
mark datasets is thus essential for both research
and industrial applications. To support this, several
evaluation frameworks–such as Language Model
Evaluation Harness (Gao et al., 2024b), Holis-
tic Evaluation of Language Models (Liang et al.,
2023)–have been developed and widely adopted by
the research community.
Despite these developments, there remains a sig-
nificant gap in the availability of comprehensive
evaluation tools tailored for RAG pipelines, where
multiple components influence overall accuracy:
(1) retrieval corpus, (2) retrieval algorithms, (3)
rerankers, (4) LLM backbones, and (5) evaluation
metrics. For instance, Magesh et al. (2024) shows
40-50% of hallucinations can originate from the
failure in document retrieval steps.
While there are existing tools (Rau et al., 2024;
Zhang et al., 2024), their utility is often limited to
general benchmark datasets and corpora, such as
MMLU (Hendrycks et al., 2021) and Wikipedia
while extending to other domain is not straightfor-arXiv:2504.01840v1  [cs.CL]  2 Apr 2025

Figure 2: System diagram
ward. Additionally, domain experts may struggle
to adapt these tools to their specific goals, as many
do not provide a graphical interface (GUI).
Here we propose LRAGE1, a holistic evalua-
tion tool explicitly designed for assessing RAG
systems in the legal domain. LRAGE extends
Language Model Evaluation Harness (Gao et al.,
2024b) by integrating it with pyserini (Lin et al.,
2021) for information retrieval, while allowing
easy control over individual components. Further-
more, LRAGE supports the use of legal-specific
corpora, such as Pile-of-Law (Henderson et al.,
2022), and benchmarks like LegalBench (Guha
et al., 2023), in an off-the-shelf manner. By provid-
ing a user-friendly GUI, LRAGE not only stream-
lines the evaluation process for legal researchers
working on RAG but also enables legal AI prac-
titioners to efficiently assess their models using
domain-specific data.
In summary, our contributions are as follows.
•We propose LRAGE , an open-source evalua-
tion tool for RAG systems that allows seam-
less integration of new corpus, tasks, and re-
trieval components.
•LRAGE features a user-friendly GUI, mak-
ing it accessibility to domain experts.
•LRAGE provides pre-configured legal
datasets for conducting RAG experiments
with ease.
2 Related work
2.1 Legal Case Retrieval
Finding relevant previous cases is critical for legal
decision-making (Feng et al., 2024). Accordingly,
1LEGAL RETRIEVAL AUGMENTED GENERATION
EVALUATION TOOLvarious studies have proposed models and datasets
to address legal retrieval tasks (Goebel et al., 2023;
Santosh et al., 2024; Hou et al., 2024; Li et al.,
2023a,b; Gao et al., 2024a; Zheng et al., 2025).
However, no comprehensive evaluation tools have
been developed to specialized in how retrieval per-
formance in legal RAG systems is influenced by
the choice of (1) retrieval corpus, (2) retreiver, (3)
backbone LLMs, (4) reranker, and (5) rubric.
2.2 RAG in legal domain
Magesh et al. (2024) analyzed commercial RAG
systems in the U.S. legal domain using 202 ex-
amples, revealing that even the most competent
system exhibited 17% hallucination rate. Niu et al.
(2024) introduced RAGTruth benchmark, built us-
ing a subset of LegalBench (Guha et al., 2023).
Their evaluation is limited to the retrieval tasks.
Zheng et al. (2025) developes two retrieval and
RAG legal benchmarks: Bar Exam QA, and Hous-
ing Statute QA based on U.S. precedents and the
statutory housing law. They show BM25 and cur-
rent dense retrievers exhibit limited performance in
recognizing gold passages in the legal domain. No-
tably, they built large scale ground truth passages
labeled by law students and legal experts.
2.3 Legal Benchmarks for LLMs
This section briefly reviews legal benchmarks de-
signed for evaluating LLMs. Guha et al. (2023) pro-
posed LegalBench, a benchmark comprising 162
legal language understanding tasks. These tasks
are organized according to six types of legal rea-
soning based on the IRAC framework. LegalBench
focuses exclusively on English legal language un-
derstanding. Kim et al. (2024b) developed KBL,
a benchmark dedicated to Korean legal language
understanding. In addition to examples, they also

Figure 3: GUI of LRAGE . It consists of six tabs: Task (top-left), Model (top-center), Generation Parameters (top-
right), Retriever (bottom-left), LLM-as-a-Judge (bottom-center), and a result tab(bottom-right). Each configuration
tab allows users to define settings, which are then used in the final tab to perform experiments and immediately
view the results.
provide resources for RAG experiments, including
a corpus of Korean statutes and precedents Hwang
et al. (2022). Fei et al. (2024) developed LawBench
comprising 20 Chinese legal tasks categorized into
three levels–Memorization, Understanding, and
Applying–based on Bloom’s taxonomy.
Except for KBL, these studies evaluated LLMs
without incorporating RAG. Also, the KBL bench-
mark utilized only a basic RAG setup, employing
a BM25 retriever without a reranker.
2.4 RAG Evaluation Tools
Rau et al. (2024) developed BERGEN, a tool de-
signed for the systematic evaluation of RAG sys-
tems in question-answering (QA) tasks. BERGEN
enables users to analyze the impact of individual
system components, offering comprehensive sup-
port for various retrievers, rerankers, and language
models. It employs an abstract class architecture
with YAML configurations, allowing users to ex-
tend and customize components according to their
requirements. Additionally, BERGEN supports
multilingual capability by providing Wikipedia in-
dices in 12 languages and offering multilingualversions of benchmark datasets.
Zhang et al. (2024) introduced RAGLAB, a
RAG evaluation tool focused on the comparative
analysis of different RAG algorithms rather than
the individual pipeline components such as retriev-
ers, rerankers, and LLMs. The framework provides
six major RAG algorithms and incorporates ten
QA benchmarks, using Wikipedia as the retrieval
corpus. RAGLAB also supports advanced evalua-
tion metrics, such as ALCE (Gao et al., 2023) and
FactScore (Min et al., 2023), for assessing gener-
ative tasks. Moreover, the frameworks allows re-
searchers to easily integrate new RAG algorithms.
Despite their strengths, these RAG evaluation
frameworks have some notable limitations. First,
these frameworks primarily rely on Wikipedia as
the sole retrieval source. In domains where spe-
cialized knowledge and domain-specific documen-
tation are critical (e.g., legal or medical fields),
this reliance is a significant limitation, as current
frameworks fail to adequately address specialized
retrieval scenarios. Second, while BERGEN pro-
vides extensibility across various pipeline compo-
nents, its flexibility comes at the cost of cumber-

some setup process, often requiring complex code
implementations and configuration files. This lack
of a no-code evaluation environment limits its ac-
cessibility and ease of use, particularly for domain
experts. In contrast, LRAGE is designed to address
these gaps. It offers seamless integration with other
retrieval corpora and tasks, a user-friendly GUI for
easy accessibility by domain experts, and off-the-
shelf legal-specific datasets, as detailed in the next
section.
3 System
LRAGE is built on top of the open-source Lan-
guage Model Evaluation Tool, lm-evaluation-
harness (Gao et al., 2024b), by incorporating Re-
triever and Reranker modules for RAG. This al-
lows us to inherit advantages such as extensibility
to various task and models and flexible system in-
struction prompt tuning.
To support various retriever and reranker frame-
works and models, LRAGE employs a modular
architecture for these components (Fig. 2). The sys-
tem follows SOLID design principles, particularly
leveraging dependency injection to ensure loosely
coupling between components, thereby enabling
high modularity and extensibility. This modular de-
sign facilitates straightforward integration of new
models and datasets without modifying the core
architecture. It defines abstract classes that specify
the essential operations for Retriever and Reranker
in constructing the RAG pipeline, which can be im-
plemented at either the framework or model level.
3.1 Retreiver and Reranker Modules
The Retriever module is currently implemented us-
ingpyserini (Lin et al., 2021), while the Reranker
module utilizes rerankers (Clavié, 2024). Both
frameworks are highly flexible and support various
models. By modularizing these components at the
framework level, LRAGE significantly reduces
the implementation overhead typically required to
support multiple models.
3.2 Metric Module
To support the evaluation of generative tasks,
we extend the metric module of lm-evaluation-
harness by integrating a custom LLM-as-a-Judge
functionality. This enables flexible, rubric-based
evaluation of legal benchmarks. By allowing
rubrics to be defined at the instance level (Min
et al., 2023; Kim et al., 2024a), LRAGE facilitatesmore detailed and precise evaluations. The users
also can easily access the evaluation results through
aggregated final scores while retaining the abil-
ity to review detailed instance-level rubric-based
assessments via stored sample logs. The module
leverages the existing LM class architecture of
lm-evaluation-harness , ensuring compatibility
with various frameworks and models that can serve
as judges.
3.3 Legal domain specialization
LRAGE offers pre-configured settings for eval-
uating RAG systems in the legal domain. It cur-
rently supports various legal benchmarks such
as KBL (Kim et al., 2024b), LegalBench (Guha
et al., 2023), and LawBench (Fei et al., 2024).
Beyond benchmark support, LRAGE provides
preprocessed resources for legal copora such as
Pile-of-Law, including a chunked version, a pre-
compiled BM25 index, and a pre-compiled FAISS
index (Douze et al., 2024), facilitating immedi-
ate use of legal datasets in RAG experiments.
The demo video for our system is available at
https://github.com/hoorangyee/LRAGE .
4 Experiments
We used Llama-3.1-8B (Meta, 2024), GPT-4 (Ope-
nAI, 2023) and various other LLMs during our
evaluations. The Pile-of-Law (Henderson et al.,
2022) corpus was chunked in a similar manner to
that described in (Hou et al., 2024). We converted
the CAIL (Xiao et al., 2018) training set into a
retrieval corpus by concatenating the fact section
with metadata. For both CAIL and Korean prece-
dents and statutes corpora (Hwang et al., 2022;
Kim et al., 2024b), we treated each individual judg-
ment as a single document for indexing, following
the setup in previous work (Kim et al., 2024b). Un-
less otherwise specified, we used Llama-3.1-8B,
BM25, and the top 3 retrieved documents as the
default setting for RAG experiments.
5 Results
To demonstrate LRAGE , we measured the overall
performance of RAG systems on legal Benchmarks
while varying the following components: retrieval
corpus, retrieval algorithm, LLM backbones, and
reranker.
Retrieval corpus We first evaluate RAG perfor-
mance on the multiple-choice questions from Ko-

Table 1: Evaluation result on 2024 Korean Bar Exam
subtasks from KBL Benchmark (Kim et al., 2024b). Ko-
rean precedents and statues (Hwang et al., 2022; Kim
et al., 2024b) (KoPS) or Korean wikipedia (kowiki)
were used as the retrieval corpus. The values in parenthe-
ses indicate the difference (in percentage points) com-
pared to the score obtained without RAG.
Acc (%, ↑) civil public criminal
Llama-3.1-8B-chat
w/o RAG 27.1 17.5 27.5
RAG w/ KoPS
BM25 28.6 (+1.5) 35.0 (+17.5) 12.5 (-15.0)
+ ColBERT reranker 27.1 (+0.0) 30.0 (+12.5) 17.5 (-10.0)
+ T5 reranker 31.4 (+4.3) 35.0 (+17.5) 15.0 (-12.5)
+ Cross-Encoder reranker 31.4 (+4.3) 40.0 (+22.5) 17.5 (-10.0)
mE5-L Dense Retrievera21.4 (-5.7) 27.5 (+10.0) 15.0 (-12.5)
bge-m3 Dense Retrieverb15.7 (-11.4) 27.5 (+10.0) 15.0 (-12.5)
RAG w/ kowiki
BM25 27.1 (+0.0) 27.5 (+10.0) 15.0 (-12.5)
+ ColBERT reranker 27.1 (+0.0) 27.5 (+10.0) 17.5 (-10.0)
+ T5 reranker 31.4 (+4.3) 25.0 (+7.5) 15.0 (-12.5)
+ Cross-Encoder reranker 27.1 (+0.0) 17.5 (+0.0) 17.5 (-10.0)
GPT-4o
w/o RAG 44.3 57.5 32.5
BM25 w/ KoPS 57.1 (+12.8) 55.0 (-2.5) 50.0 (+17.5)
a: Wang et al. (2024) b: Chen et al. (2024)
rean Bar Exam using Llama-3.1-8B and GPT-4o2
with LRAGE . With Korean precedents and stat-
ues (KoPS) as the retrieval corpus, Llama shows
improved performance compared to no RAG set-
ting in civil andpublic subtasks (Table 1, 3rd
vs. 5th–8th rows). In contrast, using kowiki cor-
pus yields little to no improvement, or in some
cases results in lower performance (3rd vs. 12th–
15th rows). We also conducted RAG evaluation for
LegalBench (Guha et al., 2023) (English), and Law-
Bench (Fei et al., 2024) (Chinese). Table 2 demon-
strates that, on three selected knowledge-intensive
subtasks from LegalBench, RAG generally im-
proves accuracy (see diagonal entries in 3rd–5th
rows), although the effectiveness still depends on
the choice of retrieval corpus. Similarly, evaluation
on three subtasks from LawBench reveals corpus-
dependent performance (Table 3). These results
align with the intuition that selecting a task-specific
corpus is crucial for effective RAG.
Retrieval algorithms In the Korean Bar Exam
experiments, the dense retriever underperforms
compared to BM25 (Table 1, 9th and 10th rows).
This suggests that domain adaptation of dense
retreivers is critical in the legal domain consis-
tent with findings from recent studies Zheng et al.
(2025); Hou et al. (2024).
2gpt-4o-2024-11-20Table 2: LegalBench (Guha et al., 2023) evaluation
result. We adopted wiki and the subsets of Pile of
Law (PoL) (Henderson et al., 2022) for the retrieval
corpus. PoL-cases includes "courtlistener_opinions",
"tax_rulings", "canadian_decisions", and "echr". PoL-
study incudes "cc_casebooks". The values in parenthe-
ses indicate the difference (in percentage points) com-
pared to w/o RAG.
Acc (%, ↑)international nys personal
citizenship judicial jurisdiction
questions ethics
w/o RAG 51.3 69.4 54.7
wiki 59.4 (+8.1) 68.7 (-0.7) 59.9 (+5.2)
PoL-cases 55.5 (+4.2) 70.9 (+1.5) 56.2 (-1.5)
PoL-study-materials 51.3 (+0.0) 69.9 (+0.5) 66.1 (+11.4)
Table 3: LawBench (Fei et al., 2024) evaluation re-
sult. Three knowledge-intensive subtasks were evalu-
ated here. 1-2: Knowledge Question Answering; 3-3:
Charge Prediction; 3-4: Preson Term Prediction w.o. Ar-
ticle. We adopted Chinese Wikipedia (zhwiki) and the
CAIL (Xiao et al., 2018) train set for the retrieval cor-
pus. The values in parentheses indicate the difference
(in percentage points) compared to the score obtained
without RAG.
LawBench1-2 3-3 3-4
ACC (%, ↑) F1 (%, ↑) -log distance ( ↑)
w/o RAG 34.4 27.2 0.59
CAIL 34.8 (+0.4) 41.2 (+14.0) 0.72 (+0.13)
zhwiki 31.8 (-2.6) 22.8 (-4.4) 0.54 (-0.05)
Reranker Next, we examine how performance
varies with the choice of rerankers. Evaluation on
thecivil andpublic subtasks shows that the cross-
encoder reranker achieves the best performance
in both cases (Table 1, 8th row). However, this
trend does not hold for the kowiki corpus, where
the T5-based reranker performs better on the civil
subtask. This demonstrates that reranker effective-
ness varies not only by task but also by corpus,
highlighting the importance of the evaluation tools
likeLRAGE which allows seamless exploration
of different RAG components combinations.
LLM backbones Interestingly, for criminal
category, both KoPS and kowiki corpora show a
significant drop in accuracy criminal task (Table 1,
3rd vs 5th–8th and 12th–15th columns). This con-
trasts with the improvements observed in stronger
API model (final two rows), suggesting that the
base capability of the model has a substantial im-
pact on RAG performance, again emphasizing the
importance of evaluation tools like LRAGE.
Further experiments with other models and sub-

Table 4: LLM-as-a-judge score on PLAT with different
rubric settings. Ssem,struc andSstrucstand for "semantic
and structural rubrics" and "structural rubrics". The
average scores from three independent experiments are
shown. GPT-4o was used as the judge model.
Model Ssem,struc Sstruc
GPT-4o-mini 4.16(±0.02) 4.41(±0.07)
GPT-4o 4.26(±0.06) 4.47(±0.07)
tasks are presented in Appendix, demonstrating
that the performance of a RAG system in the legal
domain depends on multiple interacting compo-
nents.
Rubrics We demonstrate the LLM-as-a-judge
functionality using PLAT (Choi et al., 2025), a
Korean taxation benchmark.
We first convert 50 yes/no questions about the
legitimacy of additional tax penalties from PLAT
into descriptive questions. To prepare the rubrics,
we use the reasoning section of Korean precedents
and automatically convert them into rubrics using
GPT-o1. After manually revising 10 examples in
collaborating with a tax expert, we label the remain-
ing 40 examples using GPT-o1 with a few-shot
learning approach. More details will be provided
in the paper currently in preparation.
The resulting PLAT rubrics consist of two types
of items: (1) semantic and (2) structural. Semantic
items evaluate the correctness of specific legal rea-
soning (e.g. "Is [question-specific-article]
appropriately cited?"), while structural items fo-
cus on general aspects of writing (e.g. "Is the an-
swer written concisely without unnecessary repeti-
tion?"). Each question includes four or five items,
with a total possible score of 5 points. See Ap-
pendix for more examples.
To investigate how the scores depend on the
choice of rubrics, we prepare a new set consisting
only of structural items. Since structural under-
standing may not require deep legal knowledge or
reasoning skills, less capable LLMs may achieve
similar scores with more competent LLMs. The
results show that on the original rubrics, GPT-4o-
mini achieves -0.1 score compared to GPT-4o (Ta-
ble 4, 1st column), whereas the gap narrows to
-0.06 on the new structural rubrics (final column).
We conduct additional experiments using the
BigLaw Bench core samples, an English legal task
dataset.3The rubrics comprise sixty-four 1-point
3We used examples from https://github.com/
harveyai/biglaw-bench/blob/main/blb-core/
core-samples.csv .Table 5: Evaluation results of Bar Exam QA (Zheng
et al., 2025) with agentic RAG (Roucher et al., 2025).
Model w/o RAG w/ BM25 w/ Agentic RAG
Llama-3.1-8B-Instruct 47.0 41.0 (-6.0) 47.9 (+0.9)
Llama-3.1-70B-Instruct 76.1 68.4 (-7.7) 61.5 (-14.6)
GPT-4o-mini 48.7 51.3 (+2.6) 51.3 (+2.6)
items and five 2-point items. We evaluate answers
generated by GPT-4o-mini or GPT-4o using GPT-
4o as a judge, which yields a mean score 5.63 ±0.45
(from three independent experiments). When the
point values are swapped–1-point items changed
to 2-points and vice versa–the mean score adjusts
to 5.80±0.33. Although this indicate a rise in the
average scores, it is difficult to draw a definitive
conclusions due to the limited number of examples
(five). Nevertheless, combined with the earlier ex-
periment using PLAT, the result highlights the im-
portance of supporting rubric modifications when
evaluating free-form text.
Agentic RAG To support agentic RAG, LRAGE
integrates smolagents (Roucher et al., 2025). For
the demonstration, we use recent legal RAG bench-
mark from Zheng et al. (2025). The results show
that the off-the-shelf application of agentic RAG
does not necessarily improve performance (Table 5,
2nd vs 3rd columns), although stronger model
shows relatively more competent results (1st vs
2nd rows). This suggests that, similar to retrievers,
domain adaptation of agent components–such as
prompts, tools, and reasoning frameworks (Kang
et al., 2023; Anthropic, 2025)–may be necessary.
6 Conclusion
We propose LRAGE , a holistic evaluation tool
for RAG systems specifically tailored for ap-
plications in the legal domain. Building on the
widely adapted open-source LLM evaluation tools
lm-evaluation-harness ,LRAGE integrates
two core functionalities–Retriever, Reranker–along
with additional features for evaluating generative
tasks using instance-level custom rubrics. Exper-
iments on legal domain benchmakrs demonstrate
how the overall performance of RAG systems de-
pends on individual components, highlighting the
effectiveness of LRAGE , which enables analysis
with just a few lines of script or GUI. The inclu-
sion of a user-friendly GUI and pre-processed legal
corpora for retrieval facilitates seamless adaptation
by legal domain experts, making LRAGE highly
accessible and practical for specialized use cases.

References
Anthropic. 2025. The "think" tool: Enabling claude to
stop and think in complex tool use situations. An-
thropic Engineering Blog . Accessed: 2025-03-29.
Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen,
Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi
Chen, Pei Chu, Xiaoyi Dong, Haodong Duan, Qi Fan,
Zhaoye Fei, Yang Gao, Jiaye Ge, Chenya Gu, Yuzhe
Gu, Tao Gui, Aijia Guo, Qipeng Guo, Conghui He,
Yingfan Hu, Ting Huang, Tao Jiang, Penglong Jiao,
Zhenjiang Jin, Zhikai Lei, Jiaxing Li, Jingwen Li,
Linyang Li, Shuaibin Li, Wei Li, Yining Li, Hong-
wei Liu, Jiangning Liu, Jiawei Hong, Kaiwen Liu,
Kuikun Liu, Xiaoran Liu, Chengqi Lv, Haijun Lv,
Kai Lv, Li Ma, Runyuan Ma, Zerun Ma, Wenchang
Ning, Linke Ouyang, Jiantao Qiu, Yuan Qu, Fukai
Shang, Yunfan Shao, Demin Song, Zifan Song, Zhi-
hao Sui, Peng Sun, Yu Sun, Huanze Tang, Bin Wang,
Guoteng Wang, Jiaqi Wang, Jiayu Wang, Rui Wang,
Yudong Wang, Ziyi Wang, Xingjian Wei, Qizhen
Weng, Fan Wu, Yingtong Xiong, Chao Xu, Ruil-
iang Xu, Hang Yan, Yirong Yan, Xiaogui Yang,
Haochen Ye, Huaiyuan Ying, Jia Yu, Jing Yu, Yuhang
Zang, Chuyu Zhang, Li Zhang, Pan Zhang, Peng
Zhang, Ruijie Zhang, Shuo Zhang, Songyang Zhang,
Wenjian Zhang, Wenwei Zhang, Xingcheng Zhang,
Xinyue Zhang, Hui Zhao, Qian Zhao, Xiaomeng
Zhao, Fengzhe Zhou, Zaida Zhou, Jingming Zhuo,
Yicheng Zou, Xipeng Qiu, Yu Qiao, and Dahua
Lin. 2024. Internlm2 technical report. Preprint ,
arXiv:2403.17297.
Ilias Chalkidis, Manos Fergadiotis, Prodromos Malaka-
siotis, Nikolaos Aletras, and Ion Androutsopoulos.
2020. LEGAL-BERT: The muppets straight out of
law school. In Findings of the Association for Com-
putational Linguistics: EMNLP 2020 , pages 2898–
2904, Online. Association for Computational Lin-
guistics.
Ilias Chalkidis*, Nicolas Garneau*, Catalina Goanta,
Daniel Martin Katz, and Anders Søgaard. 2023. LeX-
Files and LegalLAMA: Facilitating English Multi-
national Legal Language Model Development. In
Proceedings of the 61st Annual Meeting of the As-
sociation for Computational Linguistics , Toronto,
Canada. Association for Computational Linguistics.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint , arXiv:2402.03216.
Eunkyung Choi, Young Jin Suh, Hun Park, and Won-
seok Hwang. 2025. Taxation perspectives from large
language models: A case study on additional tax
penalties. Preprint , arXiv:2503.03444.
Benjamin Clavié. 2024. rerankers: A lightweight
python library to unify ranking methods. Preprint ,
arXiv:2408.17344.Matthew Dahl, Varun Magesh, Mirac Suzgun, and
Daniel E. Ho. 2024. Large legal fictions: Profil-
ing legal hallucinations in large language models.
Preprint , arXiv:2401.01301.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng,
Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel
Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé
Jégou. 2024. The faiss library.
Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou,
Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen,
Zhixin Yin, Zongwen Shen, Jidong Ge, and Vincent
Ng. 2024. LawBench: Benchmarking legal knowl-
edge of large language models. In Proceedings of
the 2024 Conference on Empirical Methods in Natu-
ral Language Processing , pages 7933–7962, Miami,
Florida, USA. Association for Computational Lin-
guistics.
Yi Feng, Chuanyi Li, and Vincent Ng. 2024. Legal
case retrieval: A survey of the state of the art. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers) , pages 6472–6485, Bangkok, Thailand.
Association for Computational Linguistics.
Cheng Gao, Chaojun Xiao, Zhenghao Liu, Huimin
Chen, Zhiyuan Liu, and Maosong Sun. 2024a. En-
hancing legal case retrieval via scaling high-quality
synthetic query-candidate pairs. In Proceedings of
the 2024 Conference on Empirical Methods in Natu-
ral Language Processing , pages 7086–7100, Miami,
Florida, USA. Association for Computational Lin-
guistics.
Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman,
Sid Black, Anthony DiPofi, Charles Foster, Laurence
Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li,
Kyle McDonell, Niklas Muennighoff, Chris Ociepa,
Jason Phang, Laria Reynolds, Hailey Schoelkopf,
Aviya Skowron, Lintang Sutawika, Eric Tang, An-
ish Thite, Ben Wang, Kevin Wang, and Andy Zou.
2024b. A framework for few-shot language model
evaluation.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023. Enabling large language models to generate
text with citations. In Empirical Methods in Natural
Language Processing (EMNLP) .
Randy Goebel, Yoshinobu Kano, Mi-Young Kim, Ju-
liano Rabelo, Ken Satoh, and Masaharu Yoshioka.
2023. Summary of the competition on legal infor-
mation, extraction/entailment (coliee) 2023. In Pro-
ceedings of the Nineteenth International Conference
on Artificial Intelligence and Law , ICAIL ’23, page
472–480, New York, NY , USA. Association for Com-
puting Machinery.
Neel Guha, Julian Nyarko, Daniel E. Ho, Christo-
pher Ré, Adam Chilton, Aditya Narayana, Alex
Chohlas-Wood, Austin Peters, Brandon Waldon,
Daniel N. Rockmore, Diego Zambrano, Dmitry Tal-
isman, Enam Hoque, Faiz Surani, Frank Fagan, Galit

Sarfaty, Gregory M. Dickinson, Haggai Porat, Ja-
son Hegland, Jessica Wu, Joe Nudell, Joel Niklaus,
John Nay, Jonathan H. Choi, Kevin Tobia, Mar-
garet Hagan, Megan Ma, Michael Livermore, Nikon
Rasumov-Rahe, Nils Holzenberger, Noam Kolt, Pe-
ter Henderson, Sean Rehaag, Sharad Goel, Shang
Gao, Spencer Williams, Sunny Gandhi, Tom Zur,
Varun Iyer, and Zehua Li. 2023. Legalbench: A
collaboratively built benchmark for measuring le-
gal reasoning in large language models. Preprint ,
arXiv:2308.11462.
Peter Henderson, Mark Simon Krass, Lucia Zheng,
Neel Guha, Christopher D Manning, Dan Jurafsky,
and Daniel E. Ho. 2022. Pile of law: Learning re-
sponsible data filtering from the law and a 256GB
open-source legal dataset. In Thirty-sixth Conference
on Neural Information Processing Systems Datasets
and Benchmarks Track .
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2021. Measuring massive multitask language under-
standing. Proceedings of the International Confer-
ence on Learning Representations (ICLR) .
Abe Bohan Hou, Orion Weller, Guanghui Qin, Eu-
gene Yang, Dawn Lawrie, Nils Holzenberger, An-
drew Blair-Stanek, and Benjamin Van Durme.
2024. Clerc: A dataset for legal case retrieval and
retrieval-augmented analysis generation. Preprint ,
arXiv:2406.17186.
Wonseok Hwang, Dongjun Lee, Kyoungyeon Cho,
Hanuhl Lee, and Minjoon Seo. 2022. A multi-task
benchmark for korean legal language understand-
ing and judgement prediction. In Thirty-sixth Con-
ference on Neural Information Processing Systems
Datasets and Benchmarks Track .
Xiaoxi Kang, Lizhen Qu, Lay-Ki Soon, Adnan Tra-
kic, Terry Zhuo, Patrick Emerton, and Genevieve
Grant. 2023. Can ChatGPT perform reasoning using
the IRAC method in analyzing legal scenarios like
a lawyer? In Findings of the Association for Com-
putational Linguistics: EMNLP 2023 , pages 13900–
13923, Singapore. Association for Computational
Linguistics.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. In Proceedings of the 43rd
International ACM SIGIR Conference on Research
and Development in Information Retrieval , SIGIR
’20, page 39–48, New York, NY , USA. Association
for Computing Machinery.
Seungone Kim, Juyoung Suk, Ji Yong Cho, Shayne
Longpre, Chaeeun Kim, Dongkeun Yoon, Guijin Son,
Yejin Cho, Sheikh Shafayat, Jinheon Baek, Sue Hyun
Park, Hyeonbin Hwang, Jinkyung Jo, Hyowon Cho,
Haebin Shin, Seongyun Lee, Hanseok Oh, Noah Lee,
Namgyu Ho, Se June Joo, Miyoung Ko, Yoonjoo Lee,
Hyungjoo Chae, Jamin Shin, Joel Jang, Seonghyeon
Ye, Bill Yuchen Lin, Sean Welleck, Graham Neu-
big, Moontae Lee, Kyungjae Lee, and Minjoon Seo.2024a. The biggen bench: A principled benchmark
for fine-grained evaluation of language models with
language models. Preprint , arXiv:2406.05761.
Yeeun Kim, Youngrok Choi, Eunkyung Choi, JinHwan
Choi, Hai Jin Park, and Wonseok Hwang. 2024b.
Developing a pragmatic benchmark for assessing Ko-
rean legal language understanding in large language
models. In Findings of the Association for Computa-
tional Linguistics: EMNLP 2024 , pages 5573–5595,
Miami, Florida, USA. Association for Computational
Linguistics.
Haitao Li, Qingyao Ai, Jia Chen, Qian Dong, Yueyue
Wu, Yiqun Liu, Chong Chen, and Qi Tian. 2023a.
Sailer: Structure-aware pre-trained language model
for legal case retrieval. In Proceedings of the 46th In-
ternational ACM SIGIR Conference on Research and
Development in Information Retrieval , SIGIR ’23,
page 1035–1044, New York, NY , USA. Association
for Computing Machinery.
Haitao Li, Yunqiu Shao, Yueyue Wu, Qingyao Ai, Yix-
iao Ma, and Yiqun Liu. 2023b. Lecardv2: A large-
scale chinese legal case retrieval dataset. Preprint ,
arXiv:2310.17609.
Percy Liang, Rishi Bommasani, Tony Lee, Dimitris
Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian
Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Ku-
mar, Benjamin Newman, Binhang Yuan, Bobby Yan,
Ce Zhang, Christian Alexander Cosgrove, Christo-
pher D Manning, Christopher Re, Diana Acosta-
Navas, Drew Arad Hudson, Eric Zelikman, Esin
Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren,
Huaxiu Yao, Jue WANG, Keshav Santhanam, Laurel
Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun,
Nathan Kim, Neel Guha, Niladri S. Chatterji, Omar
Khattab, Peter Henderson, Qian Huang, Ryan An-
drew Chi, Sang Michael Xie, Shibani Santurkar,
Surya Ganguli, Tatsunori Hashimoto, Thomas Icard,
Tianyi Zhang, Vishrav Chaudhary, William Wang,
Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Ko-
reeda. 2023. Holistic evaluation of language models.
Transactions on Machine Learning Research . Fea-
tured Certification, Expert Certification.
Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-
Hong Yang, Ronak Pradeep, and Rodrigo Nogueira.
2021. Pyserini: A Python toolkit for reproducible
information retrieval research with sparse and dense
representations. In Proceedings of the 44th Annual
International ACM SIGIR Conference on Research
and Development in Information Retrieval (SIGIR
2021) , pages 2356–2362.
Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suz-
gun, Christopher D. Manning, and Daniel E. Ho.
2024. Hallucination-free? assessing the reliability of
leading ai legal research tools.
Meta. 2024. The llama 3 herd of models. Preprint ,
arXiv:2407.21783.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. FActScore:
Fine-grained atomic evaluation of factual precision
in long form text generation. In Proceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing , pages 12076–12100, Singa-
pore. Association for Computational Linguistics.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu,
KaShun Shum, Randy Zhong, Juntong Song, and
Tong Zhang. 2024. RAGTruth: A hallucination cor-
pus for developing trustworthy retrieval-augmented
language models. In Proceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , pages 10862–
10878, Bangkok, Thailand. Association for Compu-
tational Linguistics.
OpenAI. 2023. Gpt-4 technical report. Preprint ,
arXiv:2303.08774.
Nicholas Pipitone and Ghita Houir Alami. 2024.
Legalbench-rag: A benchmark for retrieval-
augmented generation in the legal domain. Preprint ,
arXiv:2408.10343.
Qwen Team. 2024. Qwen2.5: A party of foundation
models.
David Rau, Hervé Déjean, Nadezhda Chirkova,
Thibault Formal, Shuai Wang, Stéphane Clinchant,
and Vassilina Nikoulina. 2024. BERGEN: A bench-
marking library for retrieval-augmented generation.
InFindings of the Association for Computational
Linguistics: EMNLP 2024 , pages 7640–7663, Mi-
ami, Florida, USA. Association for Computational
Linguistics.
LG AI Research, :, Soyoung An, Kyunghoon Bae,
Eunbi Choi, Stanley Jungkyu Choi, Yemuk Choi,
Seokhee Hong, Yeonjung Hong, Junwon Hwang,
Hyojin Jeon, Gerrard Jeongwon Jo, Hyunjik Jo,
Jiyeon Jung, Yountae Jung, Euisoon Kim, Hyosang
Kim, Joonkee Kim, Seonghwan Kim, Soyeon Kim,
Sunkyoung Kim, Yireun Kim, Youchul Kim, Ed-
ward Hwayoung Lee, Haeju Lee, Honglak Lee, Jin-
sik Lee, Kyungmin Lee, Moontae Lee, Seungjun
Lee, Woohyung Lim, Sangha Park, Sooyoun Park,
Yongmin Park, Boseong Seo, Sihoon Yang, Heuiy-
een Yeen, Kyungjae Yoo, and Hyeongu Yun. 2024.
Exaone 3.0 7.8b instruction tuned language model.
Preprint , arXiv:2408.03541.
Aymeric Roucher, Albert Villanova del Moral, Thomas
Wolf, Leandro von Werra, and Erik Kaunismäki.
2025. ‘smolagents‘: a smol library to build
great agentic systems. https://github.com/
huggingface/smolagents .
T. Y . S. S Santosh, Rashid Gustav Haddad, and Matthias
Grabmair. 2024. Ecthr-pcr: A dataset for precedent
understanding and prior case retrieval in the european
court of human rights. Preprint , arXiv:2404.00596.Gemma Team, Aishwarya Kamath, Johan Ferret,
Shreya Pathak, Nino Vieillard, Ramona Merhej,
Sarah Perrin, Tatiana Matejovicova, Alexandre
Ramé, Morgane Rivière, Louis Rouillard, Thomas
Mesnard, Geoffrey Cideron, Jean bastien Grill,
Sabela Ramos, Edouard Yvinec, Michelle Casbon,
Etienne Pot, Ivo Penchev, Gaël Liu, Francesco
Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai,
Anton Tsitsulin, Robert Busa-Fekete, Alex Feng,
Noveen Sachdeva, Benjamin Coleman, Yi Gao,
Basil Mustafa, Iain Barr, Emilio Parisotto, David
Tian, Matan Eyal, Colin Cherry, Jan-Thorsten Peter,
Danila Sinopalnikov, Surya Bhupatiraju, Rishabh
Agarwal, Mehran Kazemi, Dan Malkin, Ravin Ku-
mar, David Vilar, Idan Brusilovsky, Jiaming Luo,
Andreas Steiner, Abe Friesen, Abhanshu Sharma,
Abheesht Sharma, Adi Mayrav Gilady, Adrian
Goedeckemeyer, Alaa Saade, Alex Feng, Alexander
Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit
Vadi, András György, André Susano Pinto, Anil Das,
Ankur Bapna, Antoine Miech, Antoine Yang, Anto-
nia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal
Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie
Chen, Charline Le Lan, Christopher A. Choquette-
Choo, CJ Carey, Cormac Brick, Daniel Deutsch,
Danielle Eisenbud, Dee Cattle, Derek Cheng, Dim-
itris Paparas, Divyashree Shivakumar Sreepathi-
halli, Doug Reid, Dustin Tran, Dustin Zelle, Eric
Noland, Erwin Huizenga, Eugene Kharitonov, Fred-
erick Liu, Gagik Amirkhanyan, Glenn Cameron,
Hadi Hashemi, Hanna Klimczak-Pluci ´nska, Har-
man Singh, Harsh Mehta, Harshal Tushar Lehri,
Hussein Hazimeh, Ian Ballantyne, Idan Szpektor,
Ivan Nardini, Jean Pouget-Abadie, Jetha Chan, Joe
Stanton, John Wieting, Jonathan Lai, Jordi Orbay,
Joseph Fernandez, Josh Newlan, Ju yeong Ji, Jy-
otinder Singh, Kat Black, Kathy Yu, Kevin Hui, Ki-
ran V odrahalli, Klaus Greff, Linhai Qiu, Marcella
Valentine, Marina Coelho, Marvin Ritter, Matt Hoff-
man, Matthew Watson, Mayank Chaturvedi, Michael
Moynihan, Min Ma, Nabila Babar, Natasha Noy,
Nathan Byrd, Nick Roy, Nikola Momchev, Nilay
Chauhan, Noveen Sachdeva, Oskar Bunyan, Pankil
Botarda, Paul Caron, Paul Kishan Rubenstein, Phil
Culliton, Philipp Schmid, Pier Giuseppe Sessa, Ping-
mei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shiv-
anna, Renjie Wu, Renke Pan, Reza Rokni, Rob
Willoughby, Rohith Vallu, Ryan Mullins, Sammy
Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal,
Shashir Reddy, Shruti Sheth, Siim Põder, Sijal Bhat-
nagar, Sindhu Raghuram Panyam, Sivan Eiger, Su-
san Zhang, Tianqi Liu, Trevor Yacovone, Tyler
Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vin-
cent Roseberry, Vlad Feinberg, Vlad Kolesnikov,
Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam
Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed, Vic-
tor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao,
Kat Black, Nabila Babar, Jessica Lo, Erica Mor-
eira, Luiz Gustavo Martins, Omar Sanseviero, Lu-
cas Gonzalez, Zach Gleicher, Tris Warkentin, Va-
hab Mirrokni, Evan Senter, Eli Collins, Joelle Bar-
ral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias,
D. Sculley, Slav Petrov, Noah Fiedel, Noam Shazeer,
Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray

Kavukcuoglu, Clement Farabet, Elena Buchatskaya,
Jean-Baptiste Alayrac, Rohan Anil, Dmitry, Lep-
ikhin, Sebastian Borgeaud, Olivier Bachem, Ar-
mand Joulin, Alek Andreev, Cassidy Hardin, Robert
Dadashi, and Léonard Hussenot. 2025. Gemma 3
technical report. Preprint , arXiv:2503.19786.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024. Mul-
tilingual e5 text embeddings: A technical report.
Preprint , arXiv:2402.05672.
Chaojun Xiao, Haoxi Zhong, Zhipeng Guo, Cunchao
Tu, Zhiyuan Liu, Maosong Sun, Yansong Feng, Xian-
pei Han, Zhen Hu, Heng Wang, et al. 2018. Cail2018:
A large-scale legal dataset for judgment prediction.
arXiv preprint arXiv:1807.02478 .
Xuanwang Zhang, Yun-Ze Song, Yidong Wang, Shuyun
Tang, Xinfeng Li, Zhengran Zeng, Zhen Wu, Wei
Ye, Wenyuan Xu, Yue Zhang, Xinyu Dai, Shikun
Zhang, and Qingsong Wen. 2024. RAGLAB: A
modular and research-oriented unified framework
for retrieval-augmented generation. In Proceedings
of the 2024 Conference on Empirical Methods in Nat-
ural Language Processing: System Demonstrations ,
pages 408–418, Miami, Florida, USA. Association
for Computational Linguistics.
Lucia Zheng, Neel Guha, Javokhir Arifov, Sarah Zhang,
Michal Skreta, Christopher D. Manning, Peter Hen-
derson, and Daniel E. Ho. 2025. A reasoning-
focused legal retrieval benchmark. In Proceedings of
the 2025 Symposium on Computer Science and Law ,
CSLAW ’25, page 169–193, New York, NY , USA.
Association for Computing Machinery.

A Additional Experiments
A.1 KBL
Here, we provides additional experimental results.
Table 6 shows the results of the KBL Legal Knowl-
edge benchmark conducted using the Llama-3.1-
8B model. Similar to the result in Table 1, retriev-
ing 5 documents from KoPS consistently yields bet-
ter performance compared to the case of kowiki .
Table 6: Additional evaluation result on Legal Knowl-
edge subtasks from KBL Benchmark (Kim et al.,
2024b). Llama-3.1-8B and BM25 retriever were used.
The number indicates the average scores over 7 sub-
tasks.
Corpus Reranker Top-k=3 Top-k=5
w/o RAG 23.9
KoPS- 31.0 33.5
colbert 31.6 33.0
cross_encoder 29.4 31.8
t5 30.0 31.6
kowiki- 15.1 14.5
colbert 15.1 14.5
cross_encoder 14.9 14.9
t5 14.3 15.7
Table 7 presents the 2025 Korean Bar Exam
results with additional models.
Table 7: Evaluation results w/o RAG on 2024 Korean
Bar Exam from KBL Benchmark (Kim et al., 2024b)
with more various models.
Acc (%, ↑) civil public criminal
Gemma-3-4b-ita37.1 22.5 25.0
Gemma-3-12b-ita28.6 35.0 42.5
EXAONE-3.0-7.8B-Instructb,∗20.0 20.0 22.5
GPT-4o-mini-2024-07-18∗31.4 32.5 25.0
aTeam et al. (2025).bResearch et al. (2024).
∗Results cited from Kim et al. (2024b).
A.2 LegalBench
Here we present our initial experiments on Legal-
Bench. Table 8 presents the results of RAG ex-
periments on LegalBench-tiny with Pile-of-Law-
mini corpus. LegalBench Tiny is a subset of Legal-
Bench (Guha et al., 2023), constructed by ran-
domly sampling 10 instances per subtask, with
sampling stratified to ensure a balanced distribu-
tion of correct answers. Pile-of-Law-mini corpus
consists of 10% of randomly sampled documents
from the original corpus.
Retrieval corpus We first evaluate RAG systems
while varying their retrieval corpus: (1) no retrieval
(Table 8 1st panel), (2) Wikipedia (2nd panel), and(3) Pile-of-Law-mini (3rd panel). The result high-
light the clear importance of using domain specific
legal corpus. Interestingly, the accuracy of GPT-4o-
mini decreases the most with Pile-ofLaw-mini (3rd
panel, 4th row). To investigate this, we altered the
input order from instruction + retrieved-documents
+ examples + questions to retrieved-documents +
instruction + examples + questions and observed
an increase in accuracy (indicated by diff. prompt,
final row of each panel). We suspect this behavior
is due to the unique structure of legal documents
and GPT-4o-mini’s limited adaptation to the legal
domain.
LLM backbones Next we demonstrate how the
choice of LLM backbones, which generate the fi-
nal answers, impacts performance (Table 8). The
results reveals significant variance between mod-
els.
Table 8: Evaluation results of LegalBench-Tiny.
Model Avg Interpretation Issue Rhetorical Rule
w/o RAG
Llama3.1-8B 66.7 68.7 64.6 65.6 67.9
Qwen2.5-7B 67.7 72.5 70.5 62.9 64.7
SaulLM-7B 57.4 60.7 52.9 50.3 60.9
GPT-4o-mini 64.9 65.9 54.6 67.6 71.3
+ diff. prompt 72.7 75.2 73.3 74.2 67.9
BM25 for Wikipedia
Llama3.1-8B 67.4 (+0.7) 70.4 70.7 60.0 68.4
Qwen2.5-7B 66.3 (-1.4) 72.1 70.1 60.5 62.3
SaulLM-7B 56.3 (-1.1) 62.9 50.0 53.5 58.8
GPT-4o-mini 58.9 (-6.0) 53.0 58.0 63.2 61.4
+ diff. prompt 71.4 (-1.3) 75.3 72.1 66.9 71.1
BM25 for Pile-of-Law-mini
Llama3.1-8B 68.1 (+1.4) 69.6 71.5 63.5 67.8
Qwen2.5-7B 66.5 (-1.2) 72.7 66.6 60.4 66.1
SaulLM-7B 58.2 (+0.8) 63.3 52.9 55.6 61.1
GPT-4o-mini 55.6 (-9.3) 50.8 55.8 50.1 65.5
+ diff. prompt 73.1 (+0.4) 73.3 69.7 75.2 74.2
Retrieval algorithms Next, we examine the ef-
fect of the retrieval algorithm by replacing BM25
baseline with a dense retriever. We use LegalBERT-
base (Chalkidis et al., 2020), and LexLM-base
(Chalkidis* et al., 2023), as encoder backbone
(Table 9). Using original DPR, fine-tuend for
general-domain tasks, we observed similar perfor-
mance to BM25 (1st vs 2nd rows). However, with
domain-specialized encoders, there was a signifi-
cant improvement in accuracy (3rd and 4th rows).
When dense retriever finetuned on legal retrieval
tasks was used, performance increased further (5th
row), consistent with previous findings (Hou et al.,
2024).
We also evaluated the effect of introducing
ColBERT-based reranker(Khattab and Zaharia,

2020). Interestingly, the ColBERT reranker did
not improve performance. This result suggests that
using a reranker trained in the general domain can
reduce the accuracy of RAG system, algining with
recent findings from (Pipitone and Alami, 2024).
Table 9: Performance table under varying retrieval al-
gorithms (model fixed to GPT-4o-mini), The subset of
Pile-of-Law is used as a retrieval pool. LegalBERT-
C, LegalBERT-CR, and LegalBERT-CR GPT Stand
for "LegalBERT-DPR-CLERC", "LegalBERT-DPR-
CLERC + Reranker", "LegalBERT-DPR-CLERC +
Reranker (GPT-4o)" respectively.
Retrieval Algorithms AvgInterpretation Issue Rhetorical Rule
BM25 55.6 50.8 55.8 50.1 65.5
DPR 55.1 54.7 53.4 45.6 66.7
LegalBERT 60.4 54.7 55.6 59.0 72.3
LexLM-base 60.3 57.0 55.8 57.9 70.5
LegalBERT-C 63.7 60.0 54.2 71.4 69.0
BM25 + Reranker 55.1 50.4 54.2 51.2 64.5
LegalBERT-CR 63.5 58.7 54.8 70.9 69.7
LegalBERT-CR GPT 63.5 58.4 58.2 67.5 69.7
LexLM-base + Reranker 60.4 58.8 53.3 63.9 65.7
Experiments shown in Table 2 LegalBench also
include non-knowledge-intensive subtasks where
external documents are not required to answer the
questions. Additionally, Pile-of-Law comprises a
wide range of legal documents, many of which
may not be directely relevant. To better evalu-
ateLRAGE in a more controlled setting and to
enhance interpretability, we focus on the three
knowledge-intensive subtasks from LegalBench
and use all corresponding examples. Similarly, in-
stead of random sampling, we construct subsets of
Pile-of-Law by categorizing documents based on
type.
A.3 LawBench
Table 10 presents additional results on LawBench
for models not included in the main text. For In-
ternLM2 (Cai et al., 2024), RAG improves perfor-
mance in sections 3-3 and 3-4, but leads to lower
scores in section 1-2. In contrast, for Qwen2.5-
7B (Qwen Team, 2024), RAG improves scores in
section 1-2 but results in lower scores in sections
3-3 and 3-4.
A.4 PLAT
Table 11 presents the rubric used in the PLAT
experiment. The content was machine-translated
from Korean to English.Table 10: Evaluation of additional LLMs on Law-
Bench (Fei et al., 2024). Three knowledge-intensive
subtasks were evaluated here. 1-2: Knowledge Question
Answering; 3-3: Charge Prediction; 3-4: Preson Term
Prediction w.o. Article. We adopted Chinese Wikipedia
(zhwiki) and the CAIL (Xiao et al., 2018) train set for
the retrieval corpus.
LawBench1-2 3-3 3-4
ACC (%, ↑) F1 (%, ↑) -log distance ( ↑)
internlm2-chat-7b
w/o RAG 39.4 50.0 62.1
CAIL 24.6 52.0 74.5
zhwiki 25.8 48.0 69.0
Qwen2.5-7B-Instruct-1M
w/o RAG 29.4 56.0 75.0
CAIL 54.5 48.4 64.7
zhwiki 45.0 43.8 65.7
Table 11: Examples of rubrics used in the taxation
dataset (PLAT)
Rubric Type Content
Structural "Below are 5 evaluation criteria (total 5 points) for
the answer on ’The Legitimacy of the Penalty Tax
Imposition’ based on the above case and explanation. 1.
Structure and length of the writing (1 point): Evaluates
whether the writing follows a logical order
(introduction-main-conclusion, etc.) and is written
concisely without unnecessarily excessive length (verbose
description). 2. Formal completeness (1 point): Evaluates
whether paragraphs are divided according to the logical
flow without unnecessarily verbose expressions. 3.
Clarity of introduction and problem statement (1 point):
Whether the facts given in the case are concisely
summarized and the issue (legitimacy of penalty tax
imposition) is clearly presented. 4. Accuracy of citing
relevant laws and precedents (1 point): Evaluates whether
the laws and precedents necessary for problem-solving
such as Value Added Tax Act, Enforcement Decree,
Enforcement Rules, Framework Act on National Taxes,
precedents, etc. are appropriately cited and properly
connected to the necessary parts. 5. Adherence to
expression (1 point): Evaluates whether the case overview
and the requirements of the problem are faithfully
reflected."
Semantic and
Structural"Below are 5 evaluation criteria (total 5 points) for
the answer on ’The Legitimacy of the Penalty Tax
Imposition’ based on the above case and explanation.
Lower points are allocated to items evaluating form,
while higher points are allocated to items evaluating
content. 1. (Form) Structure and length of the writing
(0.5 points) - Whether the writing follows a logical
order (introduction-main-conclusion, etc.) - Whether it
is written concisely without unnecessarily excessive
length (verbose description) 2. (Content) Summary of
facts and presentation of main issues (1 point) - Whether
the facts appearing in the case are accurately identified
and key issues are concisely presented - Whether it
clearly emphasizes that the legitimacy of the penalty tax
imposition is at issue 3. (Content) Appropriateness of
relevant laws and interpretation (1 point) - Whether
appropriate citations are made to the Corporate Tax Act
(provisions regarding investment trusts being considered
domestic corporations), Framework Act on National Taxes,
or necessary tax law provisions - Whether it specifically
explains how these provisions can/cannot be applied to
impose penalty tax 4. (Content) Judgment of legitimate
reasons and thoroughness of argumentation (1.5 points) -
Whether the plaintiff’s argument (’investment trust is
not a taxpayer’, ’excessive refund was inevitable’) and
the defendant’s argument (’penalty tax imposition is
justified for excessive refund application’) are compared
and examined - Whether the existence of ’legitimate
reasons’ that could excuse the plaintiff from negligence
in the refund procedure is logically analyzed 5.
(Content) Validity and clarity of conclusion presentation
(1 point) - Whether a clear conclusion is drawn on
whether the penalty tax imposition is legitimate or
illegitimate - Whether the reasons supporting the
conclusion (key issues and results of legal review) are
presented concisely and clearly"