# HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks

**Authors**: Zhe Chen, Yusheng Liao, Shuyang Jiang, Zhiyuan Zhu, Haolin Li, Yanfeng Wang, Yu Wang

**Published**: 2025-08-18 09:54:10

**PDF URL**: [http://arxiv.org/pdf/2508.12778v1](http://arxiv.org/pdf/2508.12778v1)

## Abstract
Medical large vision-language Models (Med-LVLMs) have shown promise in
clinical applications but suffer from factual inaccuracies and unreliable
outputs, posing risks in real-world diagnostics. While retrieval-augmented
generation has emerged as a potential solution, current medical multimodal RAG
systems are unable to perform effective retrieval across heterogeneous sources.
The irrelevance of retrieved reports affects the factuality of analysis, while
insufficient knowledge affects the credibility of clinical decision-making. To
bridge the gap, we construct MedAtlas, which includes extensive multimodal
report repositories and diverse text corpora. Based on it, we present
HeteroRAG, a novel framework that enhances Med-LVLMs through heterogeneous
knowledge sources. The framework introduces Modality-specific CLIPs for
effective report retrieval and a Multi-corpora Query Generator for dynamically
constructing queries for diverse corpora. Incorporating knowledge from such
multifaceted sources, Med-LVLM is then trained with Heterogeneous Knowledge
Preference Tuning to achieve cross-modality and multi-source knowledge
alignment. Extensive experiments across 12 datasets and 3 modalities
demonstrate that the proposed HeteroRAG achieves state-of-the-art performance
in most medical vision language benchmarks, significantly improving factual
accuracy and reliability of Med-LVLMs.

## Full Text


<!-- PDF content starts -->

HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for
Medical Vision Language Tasks
Zhe Chen1,3, Yusheng Liao1,3, Shuyang Jiang2,3, Zhiyuan Zhu1, Haolin Li2,3
Yanfeng Wang1,3, Yu Wang1,3*
1Shanghai Jiao Tong University2Fudan University3Shanghai Artificial Intelligence Laboratory
{chenzhe2018, yuwangsjtu }@sjtu.edu.cn
Abstract
Medical large vision-language Models (Med-LVLMs) have
shown promise in clinical applications but suffer from fac-
tual inaccuracies and unreliable outputs, posing risks in real-
world diagnostics. While retrieval-augmented generation has
emerged as a potential solution, current medical multimodal
RAG systems are unable to perform effective retrieval across
heterogeneous sources. The irrelevance of retrieved reports
affects the factuality of analysis, while insufficient knowl-
edge affects the credibility of clinical decision-making. To
bridge the gap, we construct MedAtlas, which includes ex-
tensive multimodal report repositories and diverse text cor-
pora. Based on it, we present HeteroRAG, a novel framework
that enhances Med-LVLMs through heterogeneous knowl-
edge sources. The framework introduces Modality-specific
CLIPs for effective report retrieval and a Multi-corpora Query
Generator for dynamically constructing queries for diverse
corpora. Incorporating knowledge from such multifaceted
sources, Med-LVLM is then trained with Heterogeneous
Knowledge Preference Tuning to achieve cross-modality and
multi-source knowledge alignment. Extensive experiments
across 12 datasets and 3 modalities demonstrate that the pro-
posed HeteroRAG achieves state-of-the-art performance in
most medical vision language benchmarks, significantly im-
proving factual accuracy and reliability of Med-LVLMs.
1 Introduction
Large vision-language models (LVLMs) have made signifi-
cant strides in integrating multimodal information and gen-
erating natural responses (Chen et al. 2024b; Comanici et al.
2025; Bai et al. 2025). Similarly, medical LVLMs (Med-
LVLMs) show increasing promise for multimodal diagnosis
and clinical decision support (Chen et al. 2024a; Lin et al.
2025; Xu et al. 2025). However, despite these advances, cur-
rent Med-LVLMs still struggle with critical challenges re-
lated to factual accuracy and reliability (Sun et al. 2025;
Xia et al. 2025). This limitation poses serious risks in medi-
cal applications, where errors could lead to misdiagnosis or
harmful treatment recommendations.
To mitigate these limitations, recent scholarly efforts
have prioritized multimodal retrieval-augmented generation
(MMRAG) frameworks, which augment Med-LVLMs with
medical knowledge to enhance diagnostic accuracy and
*Corresponding author.
RadiologyOphthal-
mologyPathologyMedAltas
Aligned
Med-L VLMAnswerText QuestionModCLIPs
Multi-Corpora
Query Generator
 Book
Guideline
Research
Graph
WikiHetero.
Retrieval ModuleFigure 1: Overview of HeteroRAG Framework. The HRM
retrieves reports and documents from MedAtlas for a
knowledge-aligned Med-LVLM.
epistemic reliability (Ranjit et al. 2023; Sun et al. 2025;
Choi et al. 2025; Shaaban et al. 2025). Predominant method-
ologies employ multimodal retrievers, such as the medical
modality-aware CLIP models, to retrieve relevant reports us-
ing input images, owing to the strong semantic similarity be-
tween medical imaging and textual reports (Sun et al. 2025;
Xia et al. 2024, 2025). However, the training data used to
enhance the modality awareness of these retrievers is typ-
ically limited to the training splits of only a few datasets.
This constraint leads to inferior retrieval performance and ir-
relevant retrieved reports, undermining the factuality of the
Med-LVLMs. Moreover, medical corpora, such as research
articles, textbooks, and clinical guidelines, are crucial for en-
hancing the reliability of Med-LVLMs. However, the mul-
timodal retrievers fail when applied to them, as they lack
direct visual semantics and exhibit diverse linguistic charac-
teristics. Current efforts (Wu et al. 2025; Hamza et al. 2025)
attempt cross-modality document retrieval using the origi-
nal multimodal query. Though straightforward, they neglect
the alignment between queries and corpus characteristics.
One concurrent work, MIRA (Wang et al. 2025), which em-
ploys LLM-rewritten queries for improved clarity, still fails
in tailored retrieval due to limited information presented in
the rewriting prompt. In summary, current approaches fail to
perform effective retrieval across heterogeneous sources, re-
sulting in a significant knowledge gap and undermining the
factuality and credibility of medical MMRAG systems.arXiv:2508.12778v1  [cs.CL]  18 Aug 2025

A key bottleneck in addressing these limitations is the
lack of a diverse and sufficient knowledge base. To fill the
gap, we construct MedAtlas , which comprises broad mul-
timodal report repositories and rich text corpora. The report
repositories contain image-text reports in radiology, ophthal-
mology, and pathology. The text corpora are compiled from
research articles, Wikipedia entries, medical textbooks, clin-
ical guidelines, and knowledge graphs.
Building upon MedAtlas, we propose HeteroRAG, a
framework designed to significantly enhance the factual
accuracy and reliability of Med-LVLMs. As illustrated in
Figure 1, we develop the Heterogeneous Retrieval Module
(HRM), which integrates Modality-specific CLIPs (Mod-
CLIPs) and a Multi-corpora Query Generator (MQG). Mod-
CLIPs are trained on large-scale data to ensure effective
cross-modality report retrieval. The MQG module is trained
in two stages to capture corpus-specific characteristics and
generate tailored queries. Finally, we propose the Hetero-
geneous Knowledge Preference Tuning (HKPT) method to
achieve two types of alignment: (1) cross-modality align-
ment, which aligns visual inputs with retrieved textual
content; and (2) multi-source knowledge alignment, which
aligns the model’s internal knowledge with external knowl-
edge from diverse sources.
We evaluate HeteroRAG on medical visual question an-
swering and report generation tasks across 3 modalities
and 12 datasets. Empirical results show that our framework
achieves state-of-the-art performances on most benchmarks,
demonstrating its strong factuality and reliability. Notably,
HeteroRAG surpasses public Med-LVLMs, which contain
4–5× parameters, highlighting the value of effective knowl-
edge integration and alignment.
Our contributions are summarized as follows:
• We introduce MedAtlas, a newly curated comprehensive
medical database that provides rich multimodal knowl-
edge for Med-LVLMs and establishes a robust founda-
tion for medical MMRAG research.
• Leveraging MedAtlas, we propose HeteroRAG, a novel
medical MMRAG framework that performs accurate het-
erogeneous knowledge retrieval and fine-grained knowl-
edge alignment.
• Extensive experiments validate HeteroRAG’s capabil-
ity to precisely retrieve and effectively integrate multi-
source knowledge, demonstrating SOTA performance
across most benchmarks. The framework also consis-
tently outperforms substantially larger Med-LVLMs and
establishes a trustworthy and reliable foundation for
medical knowledge-intensive applications.
2 Related Work
Medical Report Retrieval for Generation. Existing
Medical MMRAG approaches primarily utilize the med-
ical images to retrieve relevant reports (He et al. 2024;
Sun et al. 2025; Xia et al. 2024, 2025). For instance,
FactMM-RAG (Sun et al. 2025) enhances report genera-
tion by incorporating high-quality reference reports. Simi-
larly, RULE (Xia et al. 2024) and MMed-RAG (Xia et al.
2025) integrate reference reports and employ preferencefine-tuning to improve model utilization of retrieved reports.
Although these approaches improve the factual accuracy of
responses, they neglect the retrieval of medical documents,
which are crucial for Med-LVLM’s reliable inference.
Medical Document Retrieval for Generation. Acknowl-
edging the limitations of report-only retrieval, recent studies
have increasingly emphasized medical documents as knowl-
edge sources (Choi et al. 2025; Shaaban et al. 2025; Wu
et al. 2025; Hamza et al. 2025). Among them, MKGF (Wu
et al. 2025) and K-LLaV A (Hamza et al. 2025) both employ
multimodal retrievers to fetch documents from the database,
aiming to mitigate hallucination issues in language mod-
els. ChatCAD+ (Zhao et al. 2024b) and MIRA (Wang et al.
2025) utilize a zero-shot query rewriting module for re-
trieval. Nevertheless, these retrieval methods overlook the
substantial content differences among various corpora, lack-
ing corpus-specific retrieval mechanisms.
3 MedAtlas Knowledge Base
The MedAtlas knowledge base comprises comprehensive
multimodal report repositories covering three modalities and
rich textual corpora from five distinct sources.
3.1 Multimodal Report Repository
Existing medical image-report repositories are limited in
scale and diversity. To address this issue, we collect image-
report pairs from a wide range of datasets. Specifically, the
Radiology subset includes 1,104,313 pairs from 6 datasets;
the Ophthalmology subset includes 111,991 pairs from 5
datasets; and the Pathology subset includes 1,514,058 pairs
from 5 datasets. To ensure data quality, duplicate pairs are
removed using the image perceptual hashing algorithm (Du,
Ho, and Cong 2020). More details are provided in Ap-
pendix B.1. For the retrieval method, we use images as
queries and reports in the library as keys to retrieve the top-k
reports, following Sun et al. (2025); Xia et al. (2024, 2025).
3.2 Textual Corpora
To ensure the richness, we collect corpora from five repre-
sentative sources. The Research corpus is drawn from the
2025 PubMed Annual Baseline. The Wiki corpus is col-
lected from the Wikipedia dumps. The Book corpus contains
e-books from PMC-LLaMA (Wu et al. 2024a), MedQA
Textbook, and StatPearls, providing foundational medical
knowledge (Xiong et al. 2024; Fan, Wang, and Liu 2025;
Chen et al. 2025). The Guideline corpus contains clini-
cal guidelines crawled from authoritative websites follow-
ing Chen et al. (2023). For the above four corpora, they are
chunked into chunks of no more than 1000 characters, with
an overlap of 200 characters following Xiong et al. (2024).
TheGraph corpus is collected from UMLS. Finally, the Re-
search, Wiki, Book, and Guideline corpora contain 51.2M,
29.7M, 14.1M, and 657.9K chunks, respectively. The Graph
corpus includes 1.7M terms and 2.9M relations.
For the retrieval of unstructured corpora, each query is
formatted as “query”, and the MedCPT models (Jin et al.
2023) are used for vector search and reranking. For the struc-
tured Graph corpus, each query is formatted as “query term,

Oph Pat
Med-L VLM MQG
ModCLIPs
Med-L VLMAligned
Med-L VLM
Rad1. Modality-awar ed CLIP  Training 2. Multi-corpora Query Generating
3. Heter ogeneous Knowledge Pr eference T uning
Book
Wiki
Research
Text Question
Guideline
Graph
Contrastive LearningQuery
Exploration
Query Judging through
Retrieved Documents 
Positive
Query SetNegative
Query Set
Preferred
QADispreferred
QASFT & DPO
DPO
Correct Answer Wrong AnswerMultiple
Queries
Report
Doc
Original Image
Original Image
Irrelevant ImageOriginal Image
Cross-Modality Alignment Knowledge Utilization Knowledge RobustnessExpert
Med-L VLM
Large-scale
CLIPBiomed-
CLIPBiomed-
CLIPBiomed-Figure 2: Overview of HeteroRAG framework. It introduces the Modality-specific CLIPs for effective report retrieval. Then, the
Multi-corpora Query Generator is developed for tailored retrieval for different corpora. Finally, HKPT is conducted to achieve
the cross-modality and multi-source knowledge alignment.
query relation”. Given the “query term”, its definition and
one-hop relationships are retrieved, followed by filtering
relevant relationships by reranking with “query relation”
(Yang et al. 2024).
4 HeteroRAG Framework
In this section, we present the HeteroRAG framework, as il-
lustrated in Figure 2. First, we introduce Modality-specific
CLIPs (ModCLIPs), which are trained on large-scale image-
text pairs for accurate report retrieval. Next, a Multi-corpora
Query Generator (MQG) is developed to enable tailored re-
trieval for multimodal questions based on corpus charac-
teristics. Finally, we propose a Heterogeneous Knowledge
Preference Tuning (HKPT) method to realize cross-modality
and multi-source knowledge alignment.
4.1 Modality-specific CLIPs
The ModCLIPs are initialized from BiomedCLIP (Zhang
et al. 2023). For each modality, the report retrieval base
is independently split into training, development, and test
sets to fine-tune CLIP models, following Xia et al. (2024,
2025). Specifically, all samples of each modality are ran-
domly split into 2000 development samples, 2000 test sam-
ples, and the remainder for training. This results in 1.10M
image-text training pairs in radiology, 0.11M in ophthalmol-
ogy, and 1.51M in pathology. Contrastive learning (Radford
et al. 2021) is performed on single-modality image-text pairs
for each ModCLIP. Compared to previous work (Sun et al.
2025; Xia et al. 2024, 2025), which relied solely on training
splits from a limited number of datasets, the significantly
scaled-up training data enables more accurate cross-modal
report retrieval.4.2 Multi-corpora Query Generator
For each multimodal question including the image vand text
question t, the module generates query set for each corpus
Q={(i, j, qi
j)|i= 1,2, ..., N C, j= 1,2, ..., Ni
q}, where
qi
jdenotes the jthquery for the ithcorpus, NCdenotes the
number of corpora, and Ni
qdenotes the number of queries
for the ithcorpus. Each query is then used to retrieve doc-
uments that collectively support answering (v, t). The train-
ing pipeline for MQG is as follows.
We begin with a query exploration phase to identify po-
tential retrieval strategies. Since annotations for documents
supporting medical multimodal questions are generally un-
available, we use the expert Med-LVLM, Lingshu-32B (Xu
et al. 2025) to generate proxy labels, as inspired by Chen
et al. (2025). Lingshu-32B consistently achieves SOTA per-
formances across most medical vision language tasks. We
prompt the expert Med-LVLM to generate multiple queries
for each source. The prompts are designed to encourage
intra-corpus diversity and align with the characteristics of
corpora. To control the cost, the number of exploration
queries per corpus is fixed to 6. Subsequently, the same ex-
pert model evaluates the documents retrieved by each query
by judging whether they support the reference answer. Man-
ual evaluation of judgment quality is conducted on a 300-
item subset by medical researchers. The results show that
Lingshu-32B achieves an accuracy of 0.837 and an F1 score
of 0.870, demonstrating the reliability of VLM-as-a-judge.
Based on the judgment, queries are categorized as either pos-
itive, denoted qw, or negative, denoted ql.
For each corpus, we select up to Ni
qinstances of qwand
qlto form positive queries Qwand negative queries Ql, re-
spectively. A two-stage training strategy is applied to MQG.
First, supervised fine-tuning (SFT) is performed:

Algorithm 1: Heterogeneous Knowledge Preference
Tuning (HKPT)
Input: D={vi, ti, Ki, yi}N
i=1: Training dataset;
K={kr, kd}: Retrieved knowledge; Mθ:
Med-LVLM; Dcm,Dmk: Preference datasets.
Output: M: Preference tuned model.
1Initialize Dcm,Dmkwith empty sets
2foreach (v, t, K, y )∈ D do
3 Retrieve the image v∗irrelevant to v
4 ▷Cross-Modality Alignment
5 ifM(v, t, K ) =yandM(v∗, t, K ) =ythen
6 xw←(v, t, K );yw←y
7 xl←(v∗, t, K );yl← M (v∗, t, K )
8 Put{xw, xl, yw, yl}intoDcm
9 ▷Multi-Source Knowledge Alignment
10 foreach k∈ {{kr},{kd},{kr, kd}}do
11 ▷Knowledge Utilization
12 ifM(v, t, K ) =yandM(v, t, K \k)̸=ythen
13 xw←(v, t, K );yw←y
14 xl←(v, t, K );yl← M (v, t, K \k)
15 Put{xw, xl, yw, yl}intoDmk
16 ▷Knowledge Robustness
17 ifM(v, t, K \k) =yandM(v, t, K )̸=ythen
18 xw←(v, t, K );yw←y
19 xl←(v, t, K );yl← M (v, t, K )
20 Put{xw, xl, yw, yl}intoDmk
21foreach (xw, xl, yw, yl)∈ Dcm∪ Dmkdo
22 Compute the loss and update Mfollowing Eq. 3
LSFT=−E(v,t,Q w)∼DwlogMθ(Qw|v, t). (1)
Then, direct preference optimization (DPO) is applied to
further align the retrieval strategies with corpora:
LDPO(Mθ;Mref) =−E(v,t,Q w,Ql)∼Dwl h
logσ
βlogMθ(Qw|v,t)
M ref(Qw|v,t)−βlogMθ(Ql|v,t)
M ref(Ql|v,t)i
.(2)
4.3 Heterogeneous Knowledge Preference Tuning
Despite retrieving relevant reports and reliable documents,
Med-LVLMs still suffer from severe knowledge misalign-
ment issues. Inspired by RULE (Xia et al. 2024) and MMed-
RAG (Xia et al. 2025), which introduce the preference fine-
tuning strategy for aligning Med-LVLMs with external re-
ports, we propose Heterogeneous Knowledge Preference
Tuning (HKPT) to enable alignment with knowledge from
more sources. The HKPT process is detailed in Algorithm 1.
Cross-Modality Alignment. The incorporation of exter-
nal knowledge may cause Med-LVLM to ignore visual in-
formation and directly copy retrieved contents (Xia et al.
2025). To mitigate this, we construct preference pairs from
the training set to improve modality alignment. Each train-
ing sample is denoted as {v, t, K, y }, where vis the medi-
cal image, tis the text question, Kis the retrieved knowl-
edge (including reports krand documents kd), and yis the
gold answer. For each v, we retrieve the least similar image
from the same modality training samples as an irrelevant im-
agev∗. Preferred responses are selected when Mcorrectlyanswers using v, while dispreferred ones are selected when
Mcorrectly answers using irrelevant v∗, indicating that M
ignores vand relies solely on K. For open-ended generation
tasks, correctness is defined as the average metric exceeding
a threshold αr. The criterion also applies below. This pro-
cess forms the preference dataset Dcm.
Multi-Source Knowledge Alignment. To improve M’s
alignment with external knowledge K, which includes re-
ports krand documents kd, we design preference pairs from
two aspects: knowledge utilization and robustness . Taking
kras an example: For knowledge utilization, preferred re-
sponses are selected when Mcorrectly answers by properly
using kr, while dispreferred ones are selected when Mfails
without kr. For knowledge robustness, preferred responses
are selected when Mcorrectly answers without kr, while
dispreferred ones are selected when Mmisuses krand pro-
duces incorrect answers. The dual-aspect strategy is also ap-
plied to kd, and a combination of krandkd, ensuring fine-
grained alignment across all knowledge sources.
The resulting Dmk, together with Dcm, are employed
in HKPT, enabling unified alignment across modalities and
knowledge sources:
LHKPT(Mθ′;Mref′) =−E(xw,xl,yw,yl)∼Dcm∪Dmk h
logσ
βlogMθ′(yw|xw)
Mref′(yw|xw)−βlogMθ′(yl|xl)
Mref′(yl|xl)i
.(3)
5 Experiments
5.1 Experimental Setups
Datasets and Metrics. The medical VQA datasets in-
clude OMVQA-Rad (Hu et al. 2024), VQA-RAD (Lau
et al. 2018), SLAKE (Liu et al. 2021), OMVQA-Oph (Hu
et al. 2024), DME-VQA (Tascon-Morales, M ´arquez-Neila,
and Sznitman 2022), Quilt-VQA (Seyfioglu et al. 2024),
PathMMU (Sun et al. 2024a), and PathVQA (He et al.
2020). Medical report generation datasets include MIMIC-
CXR (Johnson et al. 2019), IU-Xray (Demner-Fushman
et al. 2015), Harvard-FairVLMed (Luo et al. 2024), and
DeepEyeNet (Huang et al. 2021). We have carefully checked
and ensured that no sample overlap exists between these
datasets and the report database. This guarantees that the
dataset samples are unseen during ModCLIPs’ training
and prevents the retrieval results from containing in-
stances identical to the samples. Note that the perfor-
mance on Quilt-VQA can be seen as out-of-distribution
results, as the dataset does not include a training split.
Additional dataset details are provided in Appendix B.2.
For evaluation metrics, accuracy is used for closed-ended
medical VQA tasks. Radiology report generation is evalu-
ated using BLEU (Papineni et al. 2002), ROUGE-L (Lin
2004), and RaTEScore (Zhao et al. 2024a), while oph-
thalmology report generation is evaluated using BLEU,
ROUGE-L, and METEOR1(Banerjee and Lavie 2005).
Implementation Details. For the SFT and DPO training
of the MQG, we employ LoRA (Hu et al. 2022) to train the
1RaTEScore is not used for evaluating ophthalmology report
generation, as it is specifically for radiology.

Methods RetrievalRadiology Ophthalmology Pathology
OMVQA-Rad VQA-RAD SLAKE OMVQA-Oph DME-VQA Quilt-VQA PathMMU PathVQA
Original - 74.92 72.79 83.65 80.83 81.92 49.27 57.36 77.38
Beam Search - 74.25 72.43 84.86 80.58 81.92 48.40 55.85 75.97
DoLa - 74.33 73.16 84.86 80.75 81.54 50.44 55.69 77.03
VCD - 74.42 72.06 83.17 80.83 81.62 45.48 54.68 76.50
A VISC - 70.33 74.63 83.65 80.25 81.08 53.94 56.52 79.50
M3ID - 72.50 74.63 81.97 82.58 82.61 49.27 53.34 77.38
MedDr Report 78.33 74.26 83.65 83.75 81.24 53.06 62.21 77.74
FactMM-RAG Report 79.00 76.10 85.58 85.67 85.74 58.60 65.55 86.64
RULE Report 76.17 75.00 83.17 83.67 82.84 61.22 66.56 84.13
MMed-RAG Report 79.50 78.31 87.26 88.17 86.73 67.06 71.74 87.56
MKGF Doc 74.25 74.63 84.86 82.33 82.07 58.02 66.22 80.06
K-LLaV A Doc 75.83 75.00 86.30 85.25 83.22 61.81 69.23 84.02
MIAR Report+Doc 78.67 75.74 85.82 87.92 83.14 65.01 72.07 88.62
HeteroRAG (Ours) Report+Doc 82.08 80.51 86.78 91.25 83.68 69.97 75.08 91.45
Table 1: Model performance of different methods based on Lingshu-7B on the medical VQA task. The best results and second-
best results are highlighted in bold and underlined , respectively.
model initialized from Lingshu-7B. The HKPT process is
also conducted using LoRA based on Lingshu-7B. For re-
port retrieval, we adopt the adaptive retrieval context selec-
tion method (Xia et al. 2025). For document retrieval, the
MQG generates up to four queries in total for unstructured
corpora (all except Graph). Each query retrieves the top-
10 documents, which are then reranked to select the top-2
documents. For the Graph corpus, the MQG retrieves one
term and its reranked top-10 relations. The generation tem-
perature is set to 0 to ensure reproducibility. More imple-
mentation details and detailed prompts are provided in Ap-
pendix B.4 and Appendix C, respectively.
Baselines. Four categories of baselines are introduced:
(1) decoding-based methods aiming for improving factu-
ality including Beam Search (Sutskever, Vinyals, and Le
2014), DoLa (Chuang et al. 2024), VCD (Leng et al. 2024),
A VISC (Woo et al. 2024), and M3ID (Favero et al. 2024);
(2) report-retrieval methods including MedDr (He et al.
2024), FactMM-RAG (Sun et al. 2025), RULE (Xia et al.
2024), and MMed-RAG (Xia et al. 2025); (3) document-
retrieval methods including MKGF (Wu et al. 2025) and K-
LLaMA (Hamza et al. 2025) and (4) a concurrent work that
retrieves both reports and documents, MIRA (Wang et al.
2025). To ensure fair comparison, retrievable reports and
documents remain consistent across all baselines. Med-
ical CLIPs for report retrieval also remain consistent
across all baselines, with the impact of CLIP training
data analyzed separately in Section 5.3. We also introduce
widely-used Med-LVLMs: LLaV A-Med-7B (Li et al. 2023),
MedGemma-4B (Sellergren et al. 2025), HuatuoGPT-V-
34B (Chen et al. 2024a), HealthGPT-32B (Lin et al. 2025),
and Lingshu-32B (Xu et al. 2025). More baseline details are
shown in Appendix B.3.
OMVQA-Rad
VQA-RAD
SLAKE
OMVQA-Oph
DME-VQAQuilt-VQAPathMMUPathVQA
647280
667380
687786
69
79
8857
69
8056
62
68415671718089MIMIC-CXR
(RaTEScore)
IU-Xray
(RaTEScore)
Harvard-FairVLMed
(METEOR)DeepEyeNet
(METEOR)515661
515763
16
20
23162126
LLaVA-Med-7B
MedGemma-4BHuatuoGPT-V-34B
HealthGPT-32BLingshu-32B
HeteroRAG-7BFigure 3: Comparison of HeteroRAG with other Med-
LVLMs. Effective retrieval and fine-grained integration of
external knowledge enables the HeteroRAG to surpass
larger Med-LVLMs with greater parameter efficiency.
5.2 Main Results
The experimental results of different methods based on
Lingshu-7B are presented in Table 1 and Table 2. A com-
parison between widely used Med-LVLMs and HeteroRAG
is illustrated in Figure 3. These results lead to the following
key observations: (1) Effectiveness of incorporating multi-
source knowledge: HeteroRAG achieves superior perfor-
mance compared to approaches under different retrieval set-
tings. This demonstrates our effectiveness in retrieving and
integrating heterogeneous knowledge. Highly relevant re-
ports enhance the factual accuracy of Med-LVLMs, while
evidence documents improve their reliability. (2) Gener-
alizability of our framework: HeteroRAG achieves the
best performance on nearly all datasets across three modal-
ities. Notably, this superiority holds not only for closed-
ended VQA tasks but also for open-ended report generation,
which requires more sophisticated multimodal understand-
ing and generation capabilities. (3) Superiority over larger

Methods RetrievalRadiology Ophthalmology
MIMIC-CXR IU-Xray Harvard-FairVLMed DeepEyeNet
BLEU R-L RaTE BLEU R-L RaTE BLEU R-L METEOR BLEU R-L METEOR
Original - 10.31 30.39 53.30 18.50 41.00 57.95 4.21 14.30 15.75 2.35 5.06 10.20
Beam Search - 10.52 30.08 49.91 19.70 41.81 61.29 2.66 11.36 13.40 2.00 5.29 10.34
DoLa - 10.62 30.84 53.33 18.84 40.97 59.06 5.02 15.75 18.56 2.34 5.13 10.06
VCD - 11.96 27.05 49.21 19.81 35.20 56.33 7.02 11.51 14.32 2.64 4.28 9.32
A VISC - 13.52 27.43 49.01 18.86 34.80 58.75 6.94 12.59 15.80 2.52 5.90 9.14
M3ID - 11.00 28.95 51.42 17.09 35.53 56.41 7.22 13.94 16.84 2.57 6.06 9.52
MedDr Report 16.77 34.11 56.61 22.37 40.86 62.20 8.19 21.40 22.98 3.64 5.16 11.54
FactMM-RAG Report 16.82 36.22 57.20 21.82 42.69 63.22 9.15 22.56 20.76 14.91 22.22 27.08
RULE Report 17.65 34.55 56.56 19.01 38.21 59.90 8.42 21.09 16.68 14.12 20.20 25.61
MMed-RAG Report 17.65 34.84 55.72 22.40 38.96 62.67 9.89 23.27 22.10 14.36 21.36 26.36
MKGF Doc 11.56 32.33 53.66 19.97 41.32 59.64 5.87 16.18 16.83 3.44 6.65 11.55
K-LLaV A Doc 16.56 37.88 57.98 23.41 43.74 64.07 10.53 22.97 19.23 14.89 23.55 26.96
MIAR Report+Doc 17.89 37.38 58.90 22.37 42.66 63.80 10.99 23.42 22.78 14.73 23.46 26.85
HeteroRAG (Ours) Report+Doc 21.46 39.94 62.80 26.55 45.13 65.14 15.65 26.02 24.24 13.28 22.75 28.02
Table 2: Model performance of different methods based on Lingshu-7B on the medical report generation task. The best results
and second-best results are highlighted in bold and underlined , respectively.
Methods OMVQA-Rad OMVQA-Oph Quilt-VQA
Original 74.92 80.83 49.27
SFT 78.00 87.17 62.39
HeteroRAG 82.08 91.25 69.97
w/o Reports 79.17 88.92 59.77
w/o Doc 78.25 86.17 57.43
w/o Research 80.25 87.42 64.14
w/o Wiki 80.25 90.17 67.64
w/o Book 79.42 84.08 64.43
w/o Guideline 77.00 90.17 66.47
w/o Graph 81.58 88.67 69.68
Table 3: Performance of HeteroRAG after dropping each
source of knowledge.
Med-LVLMs: Figure 3 shows that HeteroRAG, with a 7B
parameter size, outperforms most advanced Med-LVLMs,
which contain 4 to 5 times more parameters across multi-
ple datasets. This indicates that the proposed framework ad-
vances the medical multimodal capabilities of existing Med-
LVLMs to a higher level.
5.3 Effectiveness of Retrieved Knowledge
We conduct ablation studies to evaluate the contribution of
knowledge sources as shown in Table 3. The “Original”
and “SFT” settings represent the performance of the orig-
inal Lingshu-7B and Lingshu-7B after SFT on the origi-
nal training set, which does not include reports and docu-
ments. The other configurations examine HeteroRAG’s per-
formance when either reports or documents are removed.
The results show that retrieved knowledge significantly im-
proves Med-LVLM’s performance compared to the Origi-
nal baseline. The performance improvements from super-Models Rad. Oph. Pat.
BiomedCLIP 30.20 13.45 28.85
PMC-CLIP 30.05 19.80 23.35
PubMedCLIP* 13.35 - -
MM-Retinal* - 4.65 -
QuiltNet* - - 39.65
FactMM-RAG* 44.25 - -
RULE* 31.80 18.90 -
MMed-RAG* 31.80 18.90 30.20
ModCLIPs* 79.40 47.55 77.35
Table 4: Image-to-text recall@5 of different retrievers. The
asterisks (*) denote the modality-specific retrievers.
vised fine-tuning alone are insufficient to compensate for
the absence of knowledge. When reports or documents
are excluded, the performance degradation confirms that
both sources are important for HeteroRAG’s knowledge-
intensive inference. Furthermore, all five corpora contribute
to Med-LVLM’s capacity.
5.4 Effectiveness of ModCLIPs
We evaluate ModCLIPs against other medical retrievers
on image-to-text report retrieval tasks, as shown in Ta-
ble 4. Generalist retrievers include BiomedCLIP and PMC-
CLIP (Lin et al. 2023). Modality-specific retrievers include
PubMedCLIP (Eslami, Meinel, and De Melo 2023), MM-
Retinal (Wu et al. 2024b), QuiltNet (Ikezogwo et al. 2023),
FactMM-RAG, RULE and MMed-RAG. Using the test set
described in Section 4.1 with recall@5 as our evaluation
metric, our experiments demonstrate that ModCLIPs con-
sistently outperform competing methods across all three
modalities. This superior performance can be attributed to

Methods OMVQA-Rad OMVQA-Oph Quilt-VQA
CLIP 76.83 84.42 63.85
MQG 82.08 91.25 69.97
w/o DPO 81.08 88.17 65.60
w/o SFT 78.75 86.00 64.14
Table 5: Performance of HeteroRAG under two ablation set-
tings: replacing MQG with CLIP-based retrieval, and re-
moving the training stages of MQG.
OriginalSFT HKPTw/o CMAw/o KU w/o KR6570758085Accuracy (↑)
OriginalSFT HKPTw/o CMAw/o KU w/o KR20304050MD (↓)
OriginalSFT HKPTw/o CMAw/o KU w/o KR50607080KUD (↓)
OriginalSFT HKPTw/o CMAw/o KU w/o KR2022242628KID (↓)
Original SFT HKPT w/o CMA w/o KU w/o KR
Figure 4: Accuracy and disalignment metrics of Lingshu-7B
trained with different methods and data.
two key advantages: (1) Single-modality training yields sig-
nificantly better modality-specific understanding compared
to mixed-modality approaches, and (2) our training data
offers more comprehensive coverage and greater diversity
within each modality.
5.5 Effectiveness of MQG
We further investigate the effectiveness of MQG in Table 5.
First, the MQG in HeteroRAG is replaced with a CLIP re-
trieval module. Specifically, for each medical visual ques-
tion, the ModCLIPs are employed to retrieve documents
through both image-to-text and text-to-text retrieval. The
two retrieval results are combined using Reciprocal Rank
Fusion. We also ablate the DPO and SFT training stages of
the MQG. The number of retrieved documents remains con-
sistent. Our experiments demonstrate that MQG retrieves
more relevant documents compared to standard CLIP meth-
ods. This improvement can be attributed to better align-
ment of MQG and each corpus’s characteristics. Further-
more, both the SFT and DPO training stages prove essential
in developing MQG.
5.6 Alignment Effectiveness of HKPT
To evaluate the alignment effectiveness of HKPT, we in-
troduce three additional metrics besides answer accuracy:
Modality Disalignment (MD), Knowledge Usage Disalign-
ment (KUD), and Knowledge Interference Disalignment
(KID). MD corresponds to CMA in Section 4.3, KUD cor-
responds to KU, and KID corresponds to KR. MD measuresModels OMVQA-Rad OMVQA-Oph Quilt-VQA
LLaV A-Med-7B 53.67 56.83 66.18
+ HeteroRAG 60.17 71.42 69.39
HuatuoGPT-V-7B 72.08 81.83 66.18
+ HeteroRAG 78.17 84.50 71.43
Lingshu-7B 74.92 80.83 49.27
+ HeteroRAG 82.08 91.25 69.97
Table 6: Model performance when the HeteroRAG frame-
work is applied to different Med-LVLMs.
the proportion that the Med-LVLM correctly answers with
the irrelevant image among cases where it correctly answers
with the original image. KUD measures the proportion that
the Med-LVLM succeeds when any retrieval source (report,
document, or report+document) is introduced among cases
where it fails without retrieval. KID measures the propor-
tion that the Med-LVLM fails when any retrieval source is
introduced among cases where it succeeds without retrieval.
Figure 4 shows the average metrics on OMVQA-Rad,
OMVQA-Oph, and Quilt-VQA. The “SFT” method refers
to SFT using the training dataset with documents and reports
added. The results demonstrate that HKPT improves over-
all accuracy compared to both the original and SFT models.
Moreover, all three types of disalignment are significantly
reduced by the HKPT method. We further conduct ablation
studies on each type of preference pair in HKPT, including
CMA, KU, and KR. The results confirm that each compo-
nent effectively enhances the corresponding alignment ca-
pability as expected.
5.7 Compatibility Analysis
To analyze the compatibility of the HeteroRAG framework
with different Med-LVLMs, we apply it to LLaV A-Med-
7B and HuatuoGPT-V-7B besides Lingshu-7B. Specifically,
the ModCLIPs and MQG in HRM are kept unchanged,
as they are universal across different downstream readers.
The HKPT process is performed separately for each Med-
LVLM. Results in Table 6 show that HeteroRAG brings con-
sistent improvements over all Med-LVLMs. This indicates
that HeteroRAG can be transferred to diverse Med-LVLMs.
6 Conclusion
This work addresses the critical challenges of effective re-
trieval and multi-aspect alignment for heterogeneous knowl-
edge in the Medical MMRAG field. MedAtlas provides a
rich, multi-source knowledge base for medical multimodal
tasks. The HeteroRAG framework enables precise report re-
trieval and multi-corpus retrieval, followed by aligning het-
erogeneous retrieval results through Heterogeneous Knowl-
edge Preference Tuning. Extensive experiments demonstrate
that our framework achieves state-of-the-art performance
across multiple medical VQA and report generation bench-
marks. Our work paves the way for effectively integrating
multi-source medical knowledge, advancing the reliability
and applicability of Med-LVLMs in clinical scenarios.

References
Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang,
K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025. Qwen2. 5-vl
technical report. arXiv preprint arXiv:2502.13923 .
Banerjee, S.; and Lavie, A. 2005. METEOR: An Automatic
Metric for MT Evaluation with Improved Correlation with
Human Judgments. In Goldstein, J.; Lavie, A.; Lin, C.; and
V oss, C. R., eds., Proceedings of the Workshop on Intrinsic
and Extrinsic Evaluation Measures for Machine Translation
and/or Summarization@ACL 2005, Ann Arbor, Michigan,
USA, June 29, 2005 , 65–72. Association for Computational
Linguistics.
Bodenreider, O. 2004. The Unified Medical Language Sys-
tem (UMLS): integrating biomedical terminology. Nucleic
Acids Res. , 32(Database-Issue): 267–270.
Chambon, P.; Delbrouck, J.-B.; Sounack, T.; Huang, S.-C.;
Chen, Z.; Varma, M.; Truong, S. Q.; Chuong, C. T.; and
Langlotz, C. P. 2024. Chexpert plus: Augmenting a large
chest x-ray dataset with text radiology reports, patient de-
mographics and additional image formats. arXiv preprint
arXiv:2405.19538 .
Chen, J.; Ouyang, R.; Gao, A.; Chen, S.; Chen, G. H.; Wang,
X.; Zhang, R.; Cai, Z.; Ji, K.; Yu, G.; Wan, X.; and Wang,
B. 2024a. HuatuoGPT-Vision, Towards Injecting Medical
Visual Knowledge into Multimodal LLMs at Scale. CoRR ,
abs/2406.19280.
Chen, Z.; Hern ´andez-Cano, A.; Romanou, A.; Bonnet, A.;
Matoba, K.; Salvi, F.; Pagliardini, M.; Fan, S.; K ¨opf, A.;
Mohtashami, A.; Sallinen, A.; Sakhaeirad, A.; Swamy, V .;
Krawczuk, I.; Bayazit, D.; Marmet, A.; Montariol, S.; Hart-
ley, M.; Jaggi, M.; and Bosselut, A. 2023. MEDITRON-
70B: Scaling Medical Pretraining for Large Language Mod-
els.CoRR , abs/2311.16079.
Chen, Z.; Liao, Y .; Jiang, S.; Wang, P.; Guo, Y .; Wang,
Y .; and Wang, Y . 2025. Towards Omni-RAG: Compre-
hensive Retrieval-Augmented Generation for Large Lan-
guage Models in Medical Applications. arXiv preprint
arXiv:2501.02460 .
Chen, Z.; Wu, J.; Wang, W.; Su, W.; Chen, G.; Xing, S.;
Zhong, M.; Zhang, Q.; Zhu, X.; Lu, L.; et al. 2024b. In-
ternvl: Scaling up vision foundation models and aligning
for generic visual-linguistic tasks. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition , 24185–24198.
Choi, K.; Yoon, B.; Kim, S.; and Park, J. 2025. Leveraging
LLMs for Multimodal Retrieval-Augmented Radiology Re-
port Generation via Key Phrase Extraction. arXiv preprint
arXiv:2504.07415 .
Chu, Y .; Zhang, K.; Malon, C.; and Min, M. R. 2025. Re-
ducing Hallucinations of Medical Multimodal Large Lan-
guage Models with Visual Retrieval-Augmented Genera-
tion. CoRR , abs/2502.15040.
Chuang, Y .; Xie, Y .; Luo, H.; Kim, Y .; Glass, J. R.; and He,
P. 2024. DoLa: Decoding by Contrasting Layers Improves
Factuality in Large Language Models. In The Twelfth In-
ternational Conference on Learning Representations, ICLR
2024, Vienna, Austria, May 7-11, 2024 . OpenReview.net.Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.;
Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang,
D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the fron-
tier with advanced reasoning, multimodality, long context,
and next generation agentic capabilities. arXiv preprint
arXiv:2507.06261 .
Decenciere, E.; Cazuguel, G.; Zhang, X.; Thibault, G.;
Klein, J.-C.; Meyer, F.; Marcotegui, B.; Quellec, G.;
Lamard, M.; Danno, R.; et al. 2013. TeleOphta: Machine
learning and image processing methods for teleophthalmol-
ogy. Irbm , 34(2): 196–203.
Demner-Fushman, D.; Kohli, M. D.; Rosenman, M. B.;
Shooshan, S. E.; Rodriguez, L.; Antani, S.; Thoma, G. R.;
and McDonald, C. J. 2015. Preparing a collection of radiol-
ogy examinations for distribution and retrieval. Journal of
the American Medical Informatics Association , 23(2): 304–
310.
Du, L.; Ho, A. T. S.; and Cong, R. 2020. Perceptual hashing
for image authentication: A survey. Signal Process. Image
Commun. , 81.
Eslami, S.; Meinel, C.; and De Melo, G. 2023. PubMed-
CLIP: How Much Does CLIP Benefit Visual Question An-
swering in the Medical Domain? In Findings of the Asso-
ciation for Computational Linguistics: EACL 2023 , 1151–
1163.
Fan, R.-Z.; Wang, Z.; and Liu, P. 2025. MegaScience: Push-
ing the Frontiers of Post-Training Datasets for Science Rea-
soning. arXiv:2507.16812.
Favero, A.; Zancato, L.; Trager, M.; Choudhary, S.; Per-
era, P.; Achille, A.; Swaminathan, A.; and Soatto, S. 2024.
Multi-Modal Hallucination Control by Visual Information
Grounding. In IEEE/CVF Conference on Computer Vision
and Pattern Recognition, CVPR 2024, Seattle, WA, USA,
June 16-22, 2024 , 14303–14312. IEEE.
Gamper, J.; and Rajpoot, N. 2021. Multiple instance cap-
tioning: Learning representations from histopathology text-
books and articles. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition , 16549–
16559.
Hamza, A.; Abdullah; Ahn, Y . H.; Lee, S.; and Kim,
S. T. 2025. LLaV A Needs More Knowledge: Retrieval
Augmented Natural Language Generation with Knowledge
Graph for Explaining Thoracic Pathologies. In Walsh, T.;
Shah, J.; and Kolter, Z., eds., AAAI-25, Sponsored by the
Association for the Advancement of Artificial Intelligence,
February 25 - March 4, 2025, Philadelphia, PA, USA , 3311–
3319. AAAI Press.
He, S.; Nie, Y .; Chen, Z.; Cai, Z.; Wang, H.; Yang, S.; and
Chen, H. 2024. MedDr: Diagnosis-Guided Bootstrapping
for Large-Scale Medical Vision-Language Learning. CoRR ,
abs/2404.15127.
He, X.; Zhang, Y .; Mou, L.; Xing, E. P.; and Xie, P. 2020.
PathVQA: 30000+ Questions for Medical Visual Question
Answering. CoRR , abs/2003.10286.
Hu, E. J.; Shen, Y .; Wallis, P.; Allen-Zhu, Z.; Li, Y .; Wang,
S.; Wang, L.; and Chen, W. 2022. LoRA: Low-Rank Adapta-
tion of Large Language Models. In The Tenth International

Conference on Learning Representations, ICLR 2022, Vir-
tual Event, April 25-29, 2022 . OpenReview.net.
Hu, Y .; Li, T.; Lu, Q.; Shao, W.; He, J.; Qiao, Y .; and Luo, P.
2024. OmniMedVQA: A New Large-Scale Comprehensive
Evaluation Benchmark for Medical LVLM. In IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
CVPR 2024, Seattle, WA, USA, June 16-22, 2024 , 22170–
22183. IEEE.
Huang, J.-H.; Yang, C.-H. H.; Liu, F.; Tian, M.; Liu, Y .-C.;
Wu, T.-W.; Lin, I.; Wang, K.; Morikawa, H.; Chang, H.; et al.
2021. Deepopht: medical report generation for retinal im-
ages via deep models and visual explanation. In Proceed-
ings of the IEEE/CVF winter conference on applications of
computer vision , 2442–2452.
Ikezogwo, W.; Seyfioglu, S.; Ghezloo, F.; Geva, D.;
Sheikh Mohammed, F.; Anand, P. K.; Krishna, R.; and
Shapiro, L. 2023. Quilt-1m: One million image-text pairs for
histopathology. Advances in neural information processing
systems , 36: 37995–38017.
Jin, D.; Pan, E.; Oufattole, N.; Weng, W.-H.; Fang, H.; and
Szolovits, P. 2020. What Disease does this Patient Have?
A Large-scale Open Domain Question Answering Dataset
from Medical Exams. arXiv preprint arXiv:2009.13081 .
Jin, Q.; Kim, W.; Chen, Q.; Comeau, D. C.; Yeganova, L.;
Wilbur, W. J.; and Lu, Z. 2023. MedCPT: Contrastive pre-
trained transformers with large-scale pubmed search logs for
zero-shot biomedical information retrieval. Bioinformatics ,
39(11): btad651.
Johnson, A. E.; Pollard, T. J.; Greenbaum, N. R.; Lungren,
M. P.; Deng, C.-y.; Peng, Y .; Lu, Z.; Mark, R. G.; Berkowitz,
S. J.; and Horng, S. 2019. MIMIC-CXR-JPG, a large pub-
licly available database of labeled chest radiographs. arXiv
preprint arXiv:1901.07042 .
Lau, J. J.; Gayen, S.; Ben Abacha, A.; and Demner-
Fushman, D. 2018. A dataset of clinically generated visual
questions and answers about radiology images. Scientific
data, 5(1): 1–10.
Leng, S.; Zhang, H.; Chen, G.; Li, X.; Lu, S.; Miao, C.; and
Bing, L. 2024. Mitigating Object Hallucinations in Large
Vision-Language Models through Visual Contrastive De-
coding. In IEEE/CVF Conference on Computer Vision and
Pattern Recognition, CVPR 2024, Seattle, WA, USA, June
16-22, 2024 , 13872–13882. IEEE.
Li, C.; Wong, C.; Zhang, S.; Usuyama, N.; Liu, H.; Yang,
J.; Naumann, T.; Poon, H.; and Gao, J. 2023. LLaV A-
Med: Training a Large Language-and-Vision Assistant for
Biomedicine in One Day. In Oh, A.; Naumann, T.; Glober-
son, A.; Saenko, K.; Hardt, M.; and Levine, S., eds., Ad-
vances in Neural Information Processing Systems 36: An-
nual Conference on Neural Information Processing Systems
2023, NeurIPS 2023, New Orleans, LA, USA, December 10
- 16, 2023 .
Li, J.; Su, T.; Zhao, B.; Lv, F.; Wang, Q.; Navab, N.; Hu,
Y .; and Jiang, Z. 2024. Ultrasound report generation with
cross-modality feature alignment via unsupervised guid-
ance. IEEE Transactions on Medical Imaging .Li, M.; Cai, W.; Liu, R.; Weng, Y .; Zhao, X.; Wang, C.;
Chen, X.; Liu, Z.; Pan, C.; Li, M.; et al. 2021. FFA-IR: To-
wards an explainable and reliable medical report generation
benchmark. In Thirty-fifth conference on neural information
processing systems datasets and benchmarks track (round
2).
Lin, C.-Y . 2004. ROUGE: A Package for Automatic Evalu-
ation of Summaries. In Text Summarization Branches Out ,
74–81. Barcelona, Spain: Association for Computational
Linguistics.
Lin, T.; Zhang, W.; Li, S.; Yuan, Y .; Yu, B.; Li, H.; He,
W.; Jiang, H.; Li, M.; Song, X.; Tang, S.; Xiao, J.; Lin, H.;
Zhuang, Y .; and Ooi, B. C. 2025. HealthGPT: A Medical
Large Vision-Language Model for Unifying Comprehension
and Generation via Heterogeneous Knowledge Adaptation.
CoRR , abs/2502.09838.
Lin, W.; Zhao, Z.; Zhang, X.; Wu, C.; Zhang, Y .; Wang, Y .;
and Xie, W. 2023. PMC-CLIP: Contrastive Language-Image
Pre-training Using Biomedical Documents. In International
Conference on Medical Image Computing and Computer-
Assisted Intervention , 525–536.
Liu, B.; Zhan, L.; Xu, L.; Ma, L.; Yang, Y .; and Wu, X.
2021. Slake: A Semantically-Labeled Knowledge-Enhanced
Dataset For Medical Visual Question Answering. In 18th
IEEE International Symposium on Biomedical Imaging,
ISBI 2021, Nice, France, April 13-16, 2021 , 1650–1654.
IEEE.
Luo, Y .; Shi, M.; Khan, M. O.; Afzal, M. M.; Huang, H.;
Yuan, S.; Tian, Y .; Song, L.; Kouhana, A.; Elze, T.; et al.
2024. FairCLIP: Harnessing fairness in vision-language
learning. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , 12289–12301.
NCBI. 2025. PubMed Baseline Data. https://ftp.ncbi.nlm.
nih.gov/pubmed/baseline/.
Papineni, K.; Roukos, S.; Ward, T.; and Zhu, W. 2002. Bleu:
a Method for Automatic Evaluation of Machine Transla-
tion. In Proceedings of the 40th Annual Meeting of the
Association for Computational Linguistics, July 6-12, 2002,
Philadelphia, PA, USA , 311–318. ACL.
Porwal, P.; Pachade, S.; Kamble, R.; Kokare, M.; Desh-
mukh, G.; Sahasrabuddhe, V .; and M ´eriaudeau, F. 2018.
Indian Diabetic Retinopathy Image Dataset (IDRiD): A
Database for Diabetic Retinopathy Screening Research.
Data , 3(3): 25.
Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.;
Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.;
Krueger, G.; and Sutskever, I. 2021. Learning Transfer-
able Visual Models From Natural Language Supervision. In
Meila, M.; and Zhang, T., eds., Proceedings of the 38th In-
ternational Conference on Machine Learning, ICML 2021,
18-24 July 2021, Virtual Event , volume 139 of Proceedings
of Machine Learning Research , 8748–8763. PMLR.
Ranjit, M. P.; Ganapathy, G.; Manuel, R.; and Ganu, T. 2023.
Retrieval Augmented Chest X-Ray Report Generation us-
ing OpenAI GPT models. In Deshpande, K.; Fiterau, M.;
Joshi, S.; Lipton, Z. C.; Ranganath, R.; Urteaga, I.; and Ye-
ung, S., eds., Machine Learning for Healthcare Conference,

MLHC 2023, 11-12 August 2023, New York, USA , volume
219 of Proceedings of Machine Learning Research , 650–
666. PMLR.
R¨uckert, J.; Bloch, L.; Br ¨ungel, R.; Idrissi-Yaghir, A.;
Sch¨afer, H.; Schmidt, C. S.; Koitka, S.; Pelka, O.; Abacha,
A. B.; G. Seco de Herrera, A.; et al. 2024. ROCOv2: Ra-
diology objects in context version 2, an updated multimodal
image dataset. Scientific Data , 11(1): 688.
Sellergren, A.; Kazemzadeh, S.; Jaroensri, T.; Kiraly, A.;
Traverse, M.; Kohlberger, T.; Xu, S.; Jamil, F.; Hughes, C.;
Lau, C.; et al. 2025. MedGemma Technical Report. arXiv
preprint arXiv:2507.05201 .
Seyfioglu, M. S.; Ikezogwo, W. O.; Ghezloo, F.; Krishna,
R.; and Shapiro, L. G. 2024. Quilt-LLaV A: Visual Instruc-
tion Tuning by Extracting Localized Narratives from Open-
Source Histopathology Videos. In IEEE/CVF Conference
on Computer Vision and Pattern Recognition, CVPR 2024,
Seattle, WA, USA, June 16-22, 2024 , 13183–13192. IEEE.
Shaaban, M. A.; Saleem, T. J.; Papineni, V . R.; and Yaqub,
M. 2025. MOTOR: Multimodal Optimal Transport via
Grounded Retrieval in Medical Visual Question Answering.
arXiv preprint arXiv:2506.22900 .
StatPearls. 2024. StatPearls. https://www.ncbi.nlm.nih.gov/
books/NBK430685/.
Sun, L.; Zhao, J. J.; Han, W.; and Xiong, C. 2025. Fact-
Aware Multimodal Retrieval Augmentation for Accurate
Medical Radiology Report Generation. In Proceedings of
the 2025 Conference of the Nations of the Americas Chapter
of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers) , 643–655.
Sun, Y .; Wu, H.; Zhu, C.; Zheng, S.; Chen, Q.; Zhang, K.;
Zhang, Y .; Wan, D.; Lan, X.; Zheng, M.; Li, J.; Lyu, X.; Lin,
T.; and Yang, L. 2024a. PathMMU: A Massive Multimodal
Expert-Level Benchmark for Understanding and Reasoning
in Pathology. In Leonardis, A.; Ricci, E.; Roth, S.; Rus-
sakovsky, O.; Sattler, T.; and Varol, G., eds., Computer Vi-
sion - ECCV 2024 - 18th European Conference, Milan, Italy,
September 29-October 4, 2024, Proceedings, Part LXII , vol-
ume 15120 of Lecture Notes in Computer Science , 56–73.
Springer.
Sun, Y .; Zhu, C.; Zheng, S.; Zhang, K.; Sun, L.; Shui, Z.;
Zhang, Y .; Li, H.; and Yang, L. 2024b. Pathasst: A genera-
tive foundation ai assistant towards artificial general intelli-
gence of pathology. In Proceedings of the AAAI Conference
on Artificial Intelligence , volume 38, 5034–5042.
Sutskever, I.; Vinyals, O.; and Le, Q. V . 2014. Sequence to
Sequence Learning with Neural Networks. In Ghahramani,
Z.; Welling, M.; Cortes, C.; Lawrence, N. D.; and Wein-
berger, K. Q., eds., Advances in Neural Information Process-
ing Systems 27: Annual Conference on Neural Information
Processing Systems 2014, December 8-13 2014, Montreal,
Quebec, Canada , 3104–3112.
Tascon-Morales, S.; M ´arquez-Neila, P.; and Sznitman, R.
2022. Consistency-Preserving Visual Question Answering
in Medical Imaging. In Wang, L.; Dou, Q.; Fletcher, P. T.;
Speidel, S.; and Li, S., eds., Medical Image Computing andComputer Assisted Intervention - MICCAI 2022 - 25th In-
ternational Conference, Singapore, September 18-22, 2022,
Proceedings, Part VIII , volume 13438 of Lecture Notes in
Computer Science , 386–395. Springer.
Tsuneki, M.; and Kanavati, F. 2022. Inference of captions
from histopathological patches. In International Confer-
ence on Medical Imaging with Deep Learning , 1235–1250.
PMLR.
Wang, J.; Ashraf, T.; Han, Z.; Laaksonen, J.; and Anwer,
R. M. 2025. MIRA: A Novel Framework for Fusing Modal-
ities in Medical RAG. arXiv preprint arXiv:2507.07902 .
Wikimedia. 2023. Wikimedia Wikipedia. https://
huggingface.co/datasets/wikimedia/wikipedia.
Woo, S.; Kim, D.; Jang, J.; Choi, Y .; and Kim, C. 2024.
Don’t Miss the Forest for the Trees: Attentional Vision
Calibration for Large Vision Language Models. CoRR ,
abs/2405.17820.
Wu, C.; Lin, W.; Zhang, X.; Zhang, Y .; Xie, W.; and Wang,
Y . 2024a. PMC-LLaMA: toward building open-source lan-
guage models for medicine. J. Am. Medical Informatics As-
soc., 31(9): 1833–1843.
Wu, R.; Zhang, C.; Zhang, J.; Zhou, Y .; Zhou, T.; and Fu, H.
2024b. MM-retinal: Knowledge-enhanced foundational pre-
training with fundus image-text expertise. In International
Conference on Medical Image Computing and Computer-
Assisted Intervention , 722–732. Springer.
Wu, Y .; Lu, Y .; Zhou, Y .; Ding, Y .; Liu, J.; and Ruan, T.
2025. MKGF: A multi-modal knowledge graph based RAG
framework to enhance LVLMs for Medical visual question
answering. Neurocomputing , 635: 129999.
Xia, P.; Zhu, K.; Li, H.; Wang, T.; Shi, W.; Wang, S.; Zhang,
L.; Zou, J.; and Yao, H. 2025. MMed-RAG: Versatile Mul-
timodal RAG System for Medical Vision Language Mod-
els. In The Thirteenth International Conference on Learning
Representations .
Xia, P.; Zhu, K.; Li, H.; Zhu, H.; Li, Y .; Li, G.; Zhang, L.;
and Yao, H. 2024. RULE: Reliable Multimodal RAG for
Factuality in Medical Vision Language Models. In Proceed-
ings of the 2024 Conference on Empirical Methods in Natu-
ral Language Processing , 1081–1093.
Xiong, G.; Jin, Q.; Lu, Z.; and Zhang, A. 2024. Benchmark-
ing Retrieval-Augmented Generation for Medicine. In Ku,
L.; Martins, A.; and Srikumar, V ., eds., Findings of the Asso-
ciation for Computational Linguistics, ACL 2024, Bangkok,
Thailand and virtual meeting, August 11-16, 2024 , 6233–
6251. Association for Computational Linguistics.
Xu, W.; Chan, H. P.; Li, L.; Aljunied, M.; Yuan, R.; Wang,
J.; Xiao, C.; Chen, G.; Liu, C.; Li, Z.; et al. 2025. Ling-
shu: A Generalist Foundation Model for Unified Multimodal
Medical Understanding and Reasoning. arXiv preprint
arXiv:2506.07044 .
Yang, R.; Liu, H.; Marrese-Taylor, E.; Zeng, Q.; Ke, Y .;
Li, W.; Cheng, L.; Chen, Q.; Caverlee, J.; Matsuo, Y .; and
Li, I. 2024. KG-Rank: Enhancing Large Language Mod-
els for Medical QA with Knowledge Graphs and Ranking
Techniques. In Demner-Fushman, D.; Ananiadou, S.; Miwa,

M.; Roberts, K.; and Tsujii, J., eds., Proceedings of the
23rd Workshop on Biomedical Natural Language Process-
ing, BioNLP@ACL 2024, Bangkok, Thailand, August 16,
2024 , 155–166. Association for Computational Linguistics.
Zhang, S.; Xu, Y .; Usuyama, N.; Xu, H.; Bagga, J.; Tinn,
R.; Preston, S.; Rao, R.; Wei, M.; Valluri, N.; et al. 2023.
BiomedCLIP: a multimodal biomedical foundation model
pretrained from fifteen million scientific image-text pairs.
arXiv preprint arXiv:2303.00915 .
Zhao, W.; Wu, C.; Zhang, X.; Zhang, Y .; Wang, Y .; and Xie,
W. 2024a. RaTEScore: A Metric for Radiology Report Gen-
eration. In Proceedings of the 2024 Conference on Em-
pirical Methods in Natural Language Processing , 15004–
15019.
Zhao, Z.; Wang, S.; Gu, J.; Zhu, Y .; Mei, L.; Zhuang, Z.;
Cui, Z.; Wang, Q.; and Shen, D. 2024b. ChatCAD+: Toward
a universal and reliable interactive CAD using LLMs. IEEE
Transactions on Medical Imaging , 43(11): 3755–3766.

A Additional Analysis
A.1 Qualitative Analysis
We provide three case studies in Figure 5. For the first case,
HeteroRAG outperforms Lingshu-7B by effectively leverag-
ing external knowledge for reasoning. It utilizes the retrieved
document’s description of “typical MRI signal characteris-
tics of fat-containing tumors” to recognize imaging features
indicative of fat content in the lesion, thereby supporting the
correct answer. Lingshu-7B lacks access to external knowl-
edge and provides an incorrect response.
For the second case, HeteroRAG outperforms Lingshu-
7B by effectively leveraging retrieved reports. It refers to key
phrases: “Low lung volumes are present,” and the impres-
sion: “Low lung volumes with probable bibasilar atelecta-
sis. No evidence of congestive heart failure.” The similar re-
ports enable clinically accurate, well-supported conclusions
for HeteroRAG.
For the third case, HeteroRAG effectively leveraged both
retrieved contents. Retrieved reports explicitly state, “in-
travascular pyogenic granulomas show a lobular growth
pattern of well-formed capillaries,” where “lobular growth
pattern” directly corresponds to the “architectural pattern”
in the question. Additionally, the retrieved documents in-
clude a research entry mentioning “infiltrating lobular car-
cinoma,” further complementing information for lobular,
which is a well-established histopathological architectural
pattern. HeteroRAG integrated this multi-source knowledge
to confirm “lobular” as the correct answer, highlighting Het-
eroRAG’s advantage in factuality and reliability.
A.2 Impact of Retrieved Report Images
Methods OMVQA-Rad OMVQA-Oph Quilt-VQA
Original 74.92 80.83 49.27
MMed-RAG 79.50 88.17 67.06
+ Report Images 76.42 86.83 63.56
HeteroRAG 82.08 91.25 69.97
+ Report Images 80.08 89.50 72.59
Table 7: Model performance when the retrieved report im-
ages are incorporated.
We further explore the integration of retrieved report im-
ages into Med-LVLMs inspired by V-RAG (Chu et al. 2025).
Specifically, we incorporate retrieved report images in con-
structing the preference pairs and training models. Results in
Table 7 indicate that adding report images does not improve
model performance and even leads to degradation on most
datasets. We attribute this to visual information in report im-
ages that is redundant with the report text, potentially hinder-
ing the model’s ability to align and integrate external knowl-
edge. Therefore, we do not include retrieved report images
in our main methods.
Question:
What is the observation in this image?
Gold Answer:
Fat-containing tumor
Retrieved Reports:
… MRI of Left and right ankle. Erosions along the medial cortex of 
the distal tibial metaphysis and epiphysis with florid periosteal 
reaction …
Retrieved Documents:
… It displays high signal on T1WI similar to subcutaneous fat, 
medium, to high signal on T2WI, and low signals on fatsuppressed
T1WI or T2WI …
Lingshu -7B: Calcified mass
HeteroRAG : Fat-containing tumor
Question:
Please generate a report for the medical image.
Gold Answer:
… Low lung volumes with probable bibasilar 
atelectasis.
Retrieved Reports:
... Low lung volumes are present … Low lung volumes with 
probable bibasilar atelectasis. No evidence of congestive heart 
failure …
Retrieved Documents:
… ityin the middle of a lung is likely to be recognized . Nodules, 
however , can be very subtle and can be single or multiple . Spotting 
the presence of nodules can make a significant difference…
Lingshu -7B: …No acute cardiopulmonary process.
HeteroRAG : … Low lung volumes with probable bibasilar 
atelectasis. No evidence of congestive heart failure.
Question:
How would you describe the architectural 
pattern of the lesion in the image?
Gold Answer:
Lobular
Retrieved Reports:
…as in their more common extravascular counterparts, 
intravascular pyogenic granulomas show a lobular growth pattern 
of well -formed capillaries …
Retrieved Documents:
… The predominant benign causes are the proliferative Aschoff
body and the main malignant cause is infiltrating lobular 
carcinoma …
Lingshu -7B: Alveolar
HeteroRAG : LobularFigure 5: Qualitative analyses for the superiority of Het-
eroRAG.

Source Modality # Pairs # Total
IU-Xray
Rad.495
1.1MPLA 14.7k
CheXpert-Plus 187.6k
MIMIC-CXR 209.6k
ROCOv2 79.8k
PMC-OA-Rad 612.2k
Harvard-FairVLMed
Oph.5.0k
112.0kDeepEyeNet 2.9k
FFA-IR 44.7k
MM-Retinal 4.4k
PMC-OA-Oph 55.1k
ARCH
Pat.6.8k
1.5MPathCap 221.3k
PatchGastric 262.8k
Quilt-1M 433.9k
PMC-OA-Pat 589.3k
Table 8: Statistics of multimodal report knowledge base in
MedAtlas.
Source Corpus # Chunks # Total
PubMed Research 51.2M 51.2M
Wikipedia Wiki 29.7M 29.7M
PMC-LLaMA
Book13.7M
14.1M MedQA 125.8k
StatPearls 322.7k
Meditron Guideline 657.9k 657.9k
- - # Terms # Relations
UMLS Graph 1.7M 2.9M
Table 9: Statistics of textual corpora in MedAtlas.
B Additional Details
B.1 MedAltas Details
The statistics of the multimodal report knowledge base and
textual corpora in MedAtlas are shown in Table 8 and
Table 9, respectively. For the multimodal report knowl-
edge base, its radiology subset includes IU-Xray (Demner-
Fushman et al. 2015), PLA (Li et al. 2024), CheXpert-
Plus (Chambon et al. 2024), MIMIC-CXR (Johnson et al.
2019), ROCOv2 (R ¨uckert et al. 2024), and PMC-OA-
Rad (Lin et al. 2023). The ophthalmology subset includes
Harvard-FairVLMed (Luo et al. 2024), DeepEyeNet (Huang
et al. 2021), FFA-IR (Li et al. 2021), MM-Retinal (Wu et al.
2024b), and PMC-OA-Oph (Lin et al. 2023). The pathology
subset includes ARCH (Gamper and Rajpoot 2021), Path-
Cap (Sun et al. 2024b), PatchGastric (Tsuneki and Kana-
vati 2022), Quilt-1M (Ikezogwo et al. 2023), and PMC-OA-
Pat (Lin et al. 2023).
The textual knowledge base of MedAtlas encompasses
a diverse collection of biomedical and general-domain cor-pora. The Research corpus includes PubMed Annual Base-
line (NCBI 2025), a comprehensive collection of biomed-
ical literature. The Wiki corpus includes Wikipedia (Wiki-
media 2023), providing broad-domain textual knowledge.
The Book corpus comprises PMC-LLaMA Books (Wu et al.
2024a), MedQA Textbooks (Jin et al. 2020), and Stat-
Pearls (StatPearls 2024), offering in-depth medical knowl-
edge from authoritative sources. The Guideline corpus in-
cludes Meditron Guidelines (Chen et al. 2023), which con-
tains curated clinical practice guidelines. The Graph corpus
is from UMLS Metathesaurus (Bodenreider 2004), a com-
prehensive semantic network that integrates concepts and re-
lationships from multiple biomedical vocabularies.
B.2 Dataset Details
The datasets used in our work include medical VQA datasets
and medical report generation datasets. The VQA datasets
are introduced as follows:
•OMVQA-Rad (Hu et al. 2024) is the radiology sub-
set of the OmniMedVQA dataset, which aggregates data
from multiple medical classification datasets and con-
verts them into a VQA format. We employ the open-
access subset. We randomly select 4,200 samples for the
training set and 1,200 samples for the test set.
•VQA-RAD (Lau et al. 2018) is the first manually curated
VQA dataset in radiology, where clinical questions were
naturally formulated by medical professionals based on
radiological images, along with reference answers. We
employ the closed-ended subset. We use the official train-
ing split of size 1,027 and the official test split of size
272.
•SLAKE (Liu et al. 2021) is a large bilingual medical
VQA dataset featuring comprehensive semantic annota-
tions by experienced physicians, accompanied by a struc-
tured medical knowledge base. We employ the English
closed-ended subset. We use the official training split of
size 1,943 and the official test split of size 416.
•OMVQA-Oph (Hu et al. 2024) is the ophthalmology
subset derived from the OmniMedVQA dataset. We em-
ploy the open-access subset. We randomly select 4,200
samples for the training set and 1,200 samples for the
test set.
•DME-VQA (Tascon-Morales, M ´arquez-Neila, and
Sznitman 2022) is built upon two public retinal image
datasets, IDRiD (Porwal et al. 2018) and e-Ophta (De-
cenciere et al. 2013), containing questions related to
diabetic macular edema (DME) and other eye conditions.
The contours of the original image masks are extracted
and rendered as red outlines on the original images to
form the question images for each sample. We randomly
select 5,000 samples from the official training split for
the training set and use the official test split of size 1,311.
•Quilt-VQA (Seyfioglu et al. 2024) is an organic eval-
uation dataset created by extracting real-world medical
questions and answers from QUILT educational videos.
We employ the closed-ended subset. We use the official
test split of size 343.

•PathMMU (Sun et al. 2024a) is a high-quality, diverse
pathology VQA dataset designed to assess the reasoning
and understanding capabilities of large multimodal mod-
els in pathology. We employ its PathCLS and Atlas sub-
sets, as they are not included in the pretraining data of
Lingshu-7B to the best of our knowledge. Then we ran-
domly select 2,095 samples for the training set and 598
samples for the test set.
•PathVQA (He et al. 2020) is the first VQA dataset in
pathology, constructed using a semi-automated pipeline
that extracts question-answer pairs from pathology text-
books and digital libraries. We employ the closed-ended
subset. We randomly select 5,000 samples from the offi-
cial training split for the training set and use the official
test split of size 3,391.
The medical report generation datasets are described as
follows:
•MIMIC-CXR (Johnson et al. 2019) is a large, publicly
available collection of chest radiographs in DICOM for-
mat, paired with free-text radiology reports from studies
conducted at the Beth Israel Deaconess Medical Center
in Boston, MA. We exclude the samples that do not con-
tain findings or impressions. We randomly select 5,000
samples from the official training split for the training set
and use the official test split of size 1,624.
•IU-Xray (Demner-Fushman et al. 2015) consists of chest
X-ray images linked to their corresponding clinical diag-
nostic reports. We exclude the samples that do not con-
tain findings or impressions. We use the official training
split of size 2,445 and the official test split of size 296.
•Harvard-FairVLMed (Luo et al. 2024) includes patient
records with SLO fundus images and clinical notes for
glaucoma diagnosis. We randomly select 3,500 samples
from the official training split for the training set and
1,000 samples from the official test split for the test set.
•DeepEyeNet (Huang et al. 2021) is a large-scale retinal
image dataset containing two modalities: grayscale fluo-
rescein angiography (FA) and color fundus photographs
(CFP), supporting various ophthalmic analysis tasks. We
randomly select 5,000 samples from the official training
split for the training set and use the official test split of
size 3,140.
B.3 Baseline Details
Decoding-based methods aiming to improve factuality are
described as follows:
•Original uses greedy decoding, which selects the token
with the highest probability at each generation step, fa-
voring locally optimal choices without considering long-
term sequence quality.
•Beam Search (Sutskever, Vinyals, and Le 2014) im-
proves upon greedy decoding by keeping track of mul-
tiple partial sequences (beams) at each step, exploring a
wider range of potential outputs and often yielding more
coherent and accurate generations.
•DoLa (Chuang et al. 2024) leverages the discrepancy be-
tween early and later layer representations in the modelby comparing their projected logits onto the vocabulary
space, guiding generation toward more accurate and con-
textually appropriate tokens.
•VCD (Leng et al. 2024) introduces a training-free decod-
ing strategy that compares outputs from original and per-
turbed visual inputs, helping to mitigate model reliance
on statistical bias and unimodal priors.
•A VISC (Woo et al. 2024) is a test-time decoding method
that enhances visual understanding by dynamically re-
calibrating attention during token generation, specifically
reducing over-attention to image tokens that lack task-
relevant content.
•M3ID (Favero et al. 2024) strengthens the impact of the
reference image during generation by amplifying tokens
that have higher mutual information with the visual input.
Medical report-retrieval methods are described as fol-
lows:
•MedDr (He et al. 2024) employs a retrieval-augmented
medical diagnosis strategy in the inference process to im-
prove the factuality of the model’s responses.
•FactMM-RAG (Sun et al. 2025) feeds the multimodal
question together with the retrieved report to the Med-
LVLM, which is fine-tuned using standard SFT to better
incorporate external reports.
•RULE (Xia et al. 2024) constructs a preference dataset
focusing on cases where over-reliance on retrieved re-
ports causes errors, aiming to balance the use of internal
knowledge and external context.
•MMed-RAG (Xia et al. 2025) extends RULE (Xia et al.
2024) by introducing cross-modality alignment to ensure
image utilization and proposing overall alignment to bet-
ter incorporate external reports.
Medical document-retrieval methods are described as fol-
lows:
•MKGF (Wu et al. 2025) uses a multimodal retriever to
fetch knowledge graphs and supplement knowledge for
LVLMs. We reproduce it using ModCLIP for image-
to-text and text-to-text retrieval to retrieve text corpora,
combining results via Reciprocal Rank Fusion.
•K-LLaV A (Hamza et al. 2025) retrieves relevant KG
triplets using a CLIP model and fine-tunes the LVLM to
incorporate the knowledge. We also use ModCLIP for re-
trieval in this method.
A concurrent work that retrieves both reports and docu-
ments is described as follows:
•MIRA (Wang et al. 2025) is a concurrent method that
retrieves both medical reports and documents. To repro-
duce it, we use the input image to retrieve similar clini-
cal cases and employ a zero-shot query-rewriting mod-
ule (Lingshu-7B) for corpus retrieval. Then the down-
stream reader is fine-tuned, whose training data includes
a chain-of-thought to guide the reader in analyzing the
external knowledge.
We also introduce widely used Med-LVLMs, which are
described as follows:

•LLaV A-Med-7B (Li et al. 2023) first aligns biomedi-
cal terminology using figure-caption pairs from scien-
tific literature, then enhances conversational understand-
ing through GPT-4-generated instruction-following data,
simulating the way non-experts gradually acquire medi-
cal knowledge through.
•MedGemma-4B (Sellergren et al. 2025) is developed
by Google and exhibits strong medical image and text
understanding capabilities, significantly outperforming
other generative models of similar size and approaching
the performance of specialized task-specific models.
•HuatuoGPT-V-34B (Chen et al. 2024a) is trained on
PubMedVision, a large-scale dataset of 1.3 million med-
ical VQA samples constructed by refining image-text
pairs from PubMed with the help of MLLMs (e.g., GPT-
4V), showing superior performance in medical multi-
modal scenarios.
•HealthGPT-32B (Lin et al. 2025) integrates medical vi-
sual comprehension and generation into a unified au-
toregressive framework, progressively adapting hetero-
geneous multimodal knowledge to a pre-trained LLM
through a bootstrapping approach.
•Lingshu-32B (Xu et al. 2025) is developed based on a
carefully curated multimodal dataset enriched with com-
prehensive medical knowledge, undergoing multi-stage
training to progressively embed domain expertise and
improve task-solving abilities, consistently outperform-
ing existing open-source models in most medical multi-
modal benchmarks.
B.4 Implementation Details
For the training of ModCLIPs, they are initialized from
BiomedCLIP (Zhang et al. 2023). The learning rate is set
to 2e-4, and the batch size is set to 512. The number of
training epochs of radiology, ophthalmology, and pathology
ModCLIP is set to 10, 100, and 10, respectively, for the dif-
ferent sizes of modality image-text pairs.
For the training of MQG, the Med-LVLM is initialized
from Lingshu-7B (Xu et al. 2025). We use LoRA (Hu et al.
2022) for efficient fine-tuning. For the SFT process, its
learning rate is set to 2e-4, the batch size is set to 64, and
the number of epochs is 3. For the DPO process, its learning
rate is set to 2e-5, the batch size is set to 64, and the number
of epochs is set to 3.
For the training of HKPT, the Med-LVLM is initialized
from Lingshu-7B. We also use LoRA (Hu et al. 2022) for
efficient fine-tuning. Its learning rate is set to 2e-5, the batch
size is set to 64, and the number of epochs is set to 4.
In our experiments, we use the development set, which
has no overlap with the training and test sets, to tune the
hyperparameters. The temperature of generation is set to 0
to ensure reproducibility. Huggingface Trainer is adopted as
the training framework for Med-LVLMs.C Prompt List
Prompt C.1: VQA with Retrieved Reports and Doc-
uments
{question image}
Retrieved Contents:
{textdoc}
Reference Reports:
{mm doc}
{question text}
Please answer the question based on the Retrieved
Contents. It should be noted that the diagnostic
information in the Reference Reports cannot be
directly used as the basis for diagnosis, but should
only be used for reference and comparison.
Answer with the option’s letter from the given
choices directly.
Prompt C.2: Report Generation with Retrieved Re-
ports and Documents
{question image}
Retrieved Contents:
{textdoc}
Reference Reports:
{mm doc}
Please answer the question based on the Retrieved
Contents. It should be noted that the diagnostic
information in the Reference Reports cannot be
directly used as the basis for diagnosis, but should
only be used for reference and comparison.
(For radiology) You are a helpful assistant. Please
generate a report for the given image, including
both findings and impressions. Return the report in
the following format: Findings: {}Impression: {}.
(For ophthalmology) You are a helpful assistant.
Please generate a short report for the given image
in 100 words. Please only include the content of the
report in your response.
Prompt C.3: Query Exploration by the Expert Med-
LVLM
{question image}
# Question (based on the image)
{question text}

# Corpus Description
research: The corpus provides access to advanced
biomedical research, facilitating access to special-
ized knowledge and resources.
wiki: The corpus provides access to general knowl-
edge across a wide range of topics.
book: The corpus provides access to medical
knowledge resource including various educational
resources and textbooks.
guideline: The corpus provides access to clinical
guidelines from leading health organizations.
graph: The corpus provides a structured knowledge
graph that connects medical definitions and related
terms.
# Query Format
<research >{query0 };{query1 }; ... (Use ; to
separate the queries) </research >
<wiki>{query0 };{query1 }; ... (Use ; to separate
the queries) </wiki>
<book>{query0 };{query1 }; ... (Use ; to separate
the queries) </book >
<guideline >{query0 };{query1 }; ... (Use ; to
separate the queries) </guideline >
<graph >{query term0},{query relation0 };
{query term1},{query relation1 }; ... (Use ; to sep-
arate the queries. Each query should use , to separate
the{query term}and{query relation })</graph >
To answer the question labeled as # Question,
please construct appropriate queries to get the
information you need.
1. Each corpus in # Corpus Description must have
search queries constructed.
2. Please give the search queries following the
format in # Query Format. Each corpus should have
6 queries, separated by ’;’.
3. The queries generated for each corpus should
exhibit diversity and be closely aligned with the
specific information needs and characteristics of
that corpus.
Prompt C.4: Query Judging through Retrieved Doc-
uments by the Expert Med-LVLM
{question image}
# Question (based on the image)
{question text}
# Gold Answer
{gold}
# Documents
{documents }You are a professional medical expert. Please
judge whether the information in the # Documents
supports the # Gold Answer as a response to the
# Question. Please judge whether # Documents
supports the # Gold Answer in response to the #
Question, rather than evaluating if the # Question’s
answer is the # Gold Answer. Please first think
step-by-step and then show your judgement using
the format <answer >yes/no </answer >at the end
of your response. Please keep your entire response
simple and complete, up to 100 words.
Prompt C.5: Query Generation by the Multi-corpora
Query Generator
{question image}
# Question (based on the image)
{question text}
# Corpus Description
research: The corpus provides access to advanced
biomedical research, facilitating access to special-
ized knowledge and resources.
wiki: The corpus provides access to general knowl-
edge across a wide range of topics.
book: The corpus provides access to medical
knowledge resource including various educational
resources and textbooks.
guideline: The corpus provides access to clinical
guidelines from leading health organizations.
graph: The corpus provides a structured knowledge
graph that connects medical definitions and related
terms.
# Query Format
<research >{query}</research >
<wiki>{query}</wiki>
<book>{query}</book >
<guideline >{query}</guideline >
<graph >{query term},{query relation }(Each
query should use , to separate the {query term}and
{query relation })</graph >
To answer the question labeled as # Question,
please construct appropriate queries to get the
information you need.
1. Please give the search queries following the
format in # Query Format. For each corpus, if you
think no information retrieval is needed, simply
output an empty tag for that corpus, for example:
<book></book >.
2. The queries generated for each corpus should be
closely aligned with the specific information needs
and characteristics of that corpus.