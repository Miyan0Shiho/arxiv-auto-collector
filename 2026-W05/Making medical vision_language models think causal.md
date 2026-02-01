# Making medical vision-language models think causally across modalities with retrieval-augmented cross-modal reasoning

**Authors**: Weiqin Yang, Haowen Xue, Qingyi Peng, Hexuan Hu, Qian Huang, Tingbo Zhang

**Published**: 2026-01-26 11:03:00

**PDF URL**: [https://arxiv.org/pdf/2601.18356v1](https://arxiv.org/pdf/2601.18356v1)

## Abstract
Medical vision-language models (VLMs) achieve strong performance in diagnostic reporting and image-text alignment, yet their underlying reasoning mechanisms remain fundamentally correlational, exhibiting reliance on superficial statistical associations that fail to capture the causal pathophysiological mechanisms central to clinical decision-making. This limitation makes them fragile, prone to hallucinations, and sensitive to dataset biases. Retrieval-augmented generation (RAG) offers a partial remedy by grounding predictions in external knowledge. However, conventional RAG depends on semantic similarity, introducing new spurious correlations. We propose Multimodal Causal Retrieval-Augmented Generation, a framework that integrates causal inference principles with multimodal retrieval. It retrieves clinically relevant exemplars and causal graphs from external sources, conditioning model reasoning on counterfactual and interventional evidence rather than correlations alone. Applied to radiology report generation, diagnosis prediction, and visual question answering, it improves factual accuracy, robustness to distribution shifts, and interpretability. Our results highlight causal retrieval as a scalable path toward medical VLMs that think beyond pattern matching, enabling trustworthy multimodal reasoning in high-stakes clinical settings.

## Full Text


<!-- PDF content starts -->

MAKING MEDICAL VISION-LANGUAGE MODELS THINK CAUSALLY
ACROSS MODALITIES WITH RETRIEV AL-AUGMENTED CROSS-MODAL REASONING
Weiqin Yang1,∗Haowen Xue2,∗,†Qingyi Peng3Hexuan Hu2Qian Huang2Tingbo Zhang2
1University of Adelaide2Hohai University3Amap
ABSTRACT
Medical vision–language models (VLMs) achieve strong
performance in diagnostic reporting and image–text align-
ment, yet their underlying reasoning mechanisms remain
fundamentally correlational, exhibiting reliance on super-
ficial statistical associations that fail to capture the causal
pathophysiological mechanisms central to clinical decision-
making. This limitation makes them fragile, prone to halluci-
nations, and sensitive to dataset biases. Retrieval-augmented
generation (RAG) offers a partial remedy by grounding pre-
dictions in external knowledge. However, conventional RAG
depends on semantic similarity, introducing new spurious
correlations. We propose Multimodal Causal Retrieval-
Augmented Generation (MCRAG), a framework that inte-
grates causal inference principles with multimodal retrieval.
MCRAG retrieves clinically relevant exemplars and causal
graphs from external sources, conditioning model reason-
ing on counterfactual and interventional evidence rather
than correlations alone. Applied to radiology report gen-
eration, diagnosis prediction, and visual question answering,
MCRAG improves factual accuracy, robustness to distribu-
tion shifts, and interpretability. Our results highlight causal
retrieval as a scalable path toward medical VLMs that think
beyond pattern matching, enabling trustworthy multimodal
reasoning in high-stakes clinical settings.
Index Terms—Vision–Language Models, Retrieval-
based Inference, Causal Inference, Multimodal Reasoning
1. INTRODUCTION
Artificial Intelligence (AI) has already transformed healthcare
and continues to hold substantial potential for further inno-
vation within clinical ecosystems. Recently, Medical Large
Vision-Language Models (Med-LVLMs) have shown great
promise for advancing interactive and intelligent diagnosis [1,
2]. Despite this potential, current Med-LVLMs still face sig-
nificant reliability issues, particularly their tendency to gen-
erate non-factual medical responses [3], making them unreli-
able in critical medical applications. These factuality issues
raise serious concerns when deploying such models in clini-
∗Equal contribution.†Corresponding author.cal settings, where even small diagnostic errors could lead to
severe consequences for patient care.
Recently, researchers have begun to focus on improving
the factuality of Med-LVLMs through various techniques, in-
cluding fine-tuning [1], low-rank adaptation [4], and retrieval-
augmented generation (RAG) [5, 6]. Fine-tuning is a direct
method to improve model performance, but faces several lim-
itations in the medical field. First, there is a lack of sufficient
high-quality labeled data for fine-tuning in the medical do-
main. Second, a distribution shift often exists between train-
ing datasets the real-world deployment data, leading to signif-
icantly worse model performance during deployment. Hence,
Retrieval-Augmented Generation (RAG) [5] has emerged as
a promising solution, grounding model outputs in external
knowledge to improve factuality. Recent works adapted RAG
for medicine, including MedRAG [6], MMed-RAG [7]. De-
spite improvements, current methods remain vulnerable to se-
mantic over-reliance, cross-modality misalignment, and spu-
rious correlations, largely due to mismatch between retrieved
contexts and visual–language grounding.
Causality-based methods seek to improve retrieval accu-
racy and representation learning, but critical limitations per-
sist [8]. CausalRAG [9] ranks contexts by causal importance,
but does not address cross-modal alignment. Similarly, CM-
CRL [10] learns shared causal representations but underuses
the structural dependencies needed for medical reasoning. Al-
though causal graphs improve interpretability [11, 12], cur-
rent approaches mainly rely on language-only causal discov-
ery [9].
These limitations necessitate a more holistic framework
that simultaneously addresses factuality and alignment by in-
tegrating causal reasoning with a multimodal structure. Ac-
cordingly, we propose to build explicit causal graphs from
multimodal data and use them to guide RAG retrieval for
medical reasoning.
In this paper, we propose MCRAG (Multimodal Causal
Retrieval-Augmented Generation), a retrieval framework de-
signed to improve factuality and robustness in Med-LVLMs.
MCRAG introduces a causal alignment graph constructed
from verifiable medical literature, capturing structured depen-
dencies between visual and textual modalities. This graph
enables retrieval guided not only by semantic similarity but
also by causal and structural relevance, thus mitigating spuri-arXiv:2601.18356v1  [cs.LG]  26 Jan 2026

Graph ConstructionImageThe bilateral pleural effusions, lower lobe volume loss …. There is mild vascular redistribution. ReportDiscover Relations
Causal Graph
VLM
Construct
CausalScoreTop-k ReportsRetrievalVLM
Generator
PromptVLM
Diagnosis
Test ImageFig. 1:MCRAG overview.Left (Graph Construction): A VLM extracts entities and relations from paired images and reports
to construct a causal graph, followed by manual refinement to prune spurious links. Right (Retrieval and Generation): For a test
image, the VLM queries the causal graph to retrieve top-k relevant reports ranked by a causal score. The retrieved reports and
test image are then combined into a prompt for the generator VLM, which produces the final diagnosis.
ous correlations. Furthermore, MCRAG incorporates RAG-
based preference fine-tuning to enforce two key principles: (i)
grounding responses in input images when relevant to prevent
degenerate text-only outputs; and (ii) interpreting retrieved
contexts causally to enhance robustness under uncertainty. Fi-
nally, MCRAG applies causal filtering to balance coverage
and precision in retrieval, selecting context based on struc-
tural importance rather than raw similarity.
Our contributions are threefold.
•MCRAG:We introduce the first framework that in-
tegrates causal graphs with cross-modal alignment for
medical vision–language generation.
•Causal alignment graph:We design a knowledge-
guided graph that enables structured cross-modal re-
trieval grounded in medical semantics.
•Preference fine-tuning:We propose a strategy that en-
forces image-grounded, causally coherent generation,
improving robustness and factual accuracy across med-
ical tasks.
2. METHODOLOGY
In this section, we presentMCRAG—a multi-modal, causal
retrieval-augmented generation framework that improves the
factuality of Med-LVLMs by tightly coupling retrieval with
an explicit Structural Causal Model (SCM). The framework
has three stages: (1)domain-aware retrieval, which selects
the optimal retriever for each input; (2)adaptive context se-
lection, which filters and sizes evidence on the fly; and (3)RAG-based preference fine-tuning, which aligns responses
with SCM-supported evidence.
2.1. Structural Causal Model (SCM)
At the core of the MCRAG framework is the formal repre-
sentation of medical knowledge as aStructural Causal Model
(SCM), a mathematical formalism for causal inference. An
SCM, denoted asM, is a tuple⟨V,U,F⟩, where:
•Vis a set of endogenous variables, representing the
manifest, observable variables within the system. In
the medical context,Vincludes variables correspond-
ing to image regions (V I), clinical findings (V F), pa-
tient symptoms (V S), and diagnostic outcomes (V D).
•Uis a set of exogenous variables, representing latent or
unobserved factors. These variables account for all fac-
tors influencing the endogenous variables that are not
explicitly included in the model, such as genetic predis-
positions or data heterogeneity across hospital systems.
•Fis a set of structural equations, one for each variable
Vi∈V. Each equationf i∈ Fdefines the value ofV i
as a function of its parents,Pa(V i), in the causal graph
and its corresponding exogenous variableU i∈U. For
instance, a function for a specific clinical finding might
be expressed as:
ˆVF=fF(Pa(V F), UF)(1)
A key feature of the SCM is its associated causal graph
G, a directed graph over the variables inVandU.

2.2. Cross-Modal Medical Causal Graph Construction
The construction of a comprehensive causal graphGre-
quires integrating information from multiple modalities. Our
framework employs a two-stage data-driven causal discovery
protocol to buildGfrom a corpus of paired medical images
and clinical reports.
Step 1: Multimodal VLMs-Assisted Causal Discovery.
We use vision-Language Models (VLMs) to serve as the pri-
mary knowledge extractor. The model is prompted to analyze
image-text pairs to identify potential causal relationships.
For instance, a visual feature like‘pulmonary opacity’ob-
served in a chest X-ray would be linked to the textual entity
‘pneumonia’in the accompanying report, proposing a causal
edge between them, grounding textual concepts in visual
evidence. Letv iandr jdenote the visual embedding of im-
ageiand textual report embedding of reportj, respectively.
The retriever’s contrastive lossLretrmaximizes the cosine
similaritys ij=⟨v i, rj⟩for true image–report pairs while
minimizings ijfor mismatched pairs. In practice, we collect
a corpus of domain-specific text (e.g. reports for radiology
images) and use these as the knowledge base.
Step 2: Manual Graph Refinement.Starting from the draft
graph proposed by the VLMs under a low-confidence thresh-
old, we conduct a principled manual review of every candi-
date causal edge. Each edge is evaluated for clinical plau-
sibility and statistical support (e.g., whether the conditional
probability of a diagnosis given a visual feature corresponds
with domain knowledge), and any edge failing this inspection
is removed. For instance, if a visual featureV Iand the final
diagnosisV Dare conditionally independent given a textual
clinical findingV F(i.e.,V I⊥ ⊥V D|VF), this provides sta-
tistical evidence for the causal pathwayV I→VF→VDand
justifies pruning the spurious direct edgeV I→V D. Clini-
cally unreasonable edges are discarded even if strong statisti-
cal associations appear.
2.3. Causal-based Retrieval Augmented Reasoning
Given an input imageI, we first retrieve the top-Knearest
textual reports{R k}K
k=1in a joint embedding space. We then
enforce causal consistency using the graphG. For each candi-
dateR k, we extract the variables it references (e.g., findings
VFand diagnosesV D) and evaluate how well they are sup-
ported by image-derived featuresV Ialong the causal paths in
G(preferablyV I→VF→VD).
Score(R k) = (1−α) logp G 
VD, VF|VI
+αsim(I, R k), α∈[0,1](2)
wheresim(I, R k)denotes the image–report embedding simi-
larity, andp G(·)is the likelihood induced by the factorizationimplied by the causal graphG. For example, ifGretains the
mediated pathV I→VF→VD, then
pG(VD, VF|VI) =p(V F|VI)p(V D|VF)(3)
Candidates consistent withGare up-weighted, whereas
those relying on unsupported or pruned edges are down-
weighted or discarded, yielding retrieved reports that are both
semantically relevant and causally grounded.
After assembling high-quality retrieved contexts and their
associated causal relations, MCRAG integrates them into the
generation process via retrieval-augmented fine-tuning.
3. EXPERIMENT
3.1. Experimental Setups
For the language model, we adopt LLaV A-Med-1.5-7B [1],
fine-tuned with LoRA [19] using the AdamW optimizer. The
fine-tuning is performed with a learning rate of3×10−5,
weight decay of10−2, a batch size of 16, and for 500 epochs.
For modality-specific encoders, we employ MedVIT [20] as
the vision encoder and BioClinicalBERT [21] as the text en-
coder.
We adopt the experimental framework of MMed-RAG[7]
and evaluate hallucination mitigation methods from two com-
plementary perspectives. Decoding-based approaches, such
as DoLa [13], OPERA [14], and VCD [15], improve factual
consistency by directly adjusting the model’s output distribu-
tion. In contrast, multimodal retrieval-augmented generation
(RAG) methods, including MedDr [16], FactMM-RAG [17],
RULE [18], and MMed-RAG [7], mitigate hallucinations
by grounding responses in external knowledge. We didn’t
choose the CasualRAG is because it is not multi-modal, so
not in our scope.
Our experiments employ MIMIC-CXR [22] and IU-
Xray [23] as benchmark datasets. Question–answer pairs
are taken from MMed-RAG [7]. Following prior work [7],
we assess medical VQA performance using Accuracy, F1
Score, and AUROC, while report generation is evaluated with
BLEU, ROUGE-L, and METEOR.
3.2. Comparison Results
Table 1 compares decoding-only baselines with retrieval-
augmented models. While MMed-RAG delivers strong re-
sults (e.g.,89.54Acc and87.13AUC on IU-Xray VQA),
our method (MCRAG) consistently sets new state-of-the-
art across all tasks. On IU-Xray VQA, MCRAGsurpasses
MMed-RAG by+0.58Acc,+1.31F1, and+1.12AUC;
on MIMIC-CXR VQA, it achieves further gains of+1.34
Acc,+0.88F1, and+1.34AUC. For report generation,
MCRAGraises BLEU to35.02and25.81, improving over
MMed-RAG by+3.64and+2.56on IU-Xray and MIMIC-
CXR, respectively. These results demonstrate that causality-
guided retrieval not only enhances factual accuracy in VQA

Table 1: Performance (%) of different methods on Radiology VQA and Radiology Report Generation. For VQA, we report
Accuracy, F1 score, and AUROC; for Report Generation, we report BLEU, ROUGE-L (R-L), and METEOR (MET). The best
and second-best results are highlighted in red and blue , respectively. Comparison results are reported from MMed-RAG [7].
ModelsRadiology VQA Radiology Report Generation
IU-Xray MIMIC-CXR IU-Xray MIMIC-CXR
Acc↑F1↑AUC↑ Acc↑F1↑AUC↑ BLEU↑R-L↑MET↑ BLEU↑R-L↑MET↑
LLaV A-Med-1.5 [1] 75.47 64.04 67.46 75.79 80.49 68.84 9.64 12.26 8.21 12.11 13.05 11.16
+ DoLa [13] 78.00 66.75 72.19 81.35 85.73 72.73 11.79 15.82 12.72 17.11 14.89 14.81
+ OPERA [14] 70.59 61.54 63.22 69.34 76.66 62.46 10.66 14.70 12.01 15.40 12.52 13.72
+ VCD [15] 68.99 54.35 61.08 70.89 75.57 64.61 10.42 14.14 11.59 15.18 12.30 13.38
+ MedDr [16] 83.33 67.80 77.15 55.16 56.18 58.47 12.37 16.45 13.50 18.59 15.72 16.77
+ FactMM-RAG [17] 84.51 68.51 77.07 77.58 81.86 70.09 14.70 18.05 15.92 18.71 15.84 16.82
+ RULE [18] 87.84 78.00 85.78 83.92 87.49 83.44 27.53 23.16 27.99 18.61 15.96 17.42
+ MMed-RAG [7] 89.54 80.72 87.13 83.57 88.49 85.08 31.38 25.59 32.43 23.25 12.34 20.47
+ MCRAG 90.12 82.03 88.25 84.91 89.37 86.42 35.02 28.47 35.18 25.81 15.05 22.34
Table 2: Ablation study of causality in MCRAG on the
MIMIC-CXR dataset. We report Accuracy (Acc), F1, and
BLEU as mean±std over 3 runs.τdenotes the ratio of causal
branches removed for refining causal links.
Method Acc↑F1↑BLEU↑
MCRAG(Full Model,τ= 0.7) 84.91±0.21 89.37±0.18 25.81±0.42
w/o Causality Relation 81.26±0.33 86.71±0.29 23.58±0.55
w/o Manual Refining 80.34±0.27 85.42±0.31 22.47±0.61
τ= 0.5 83.47±0.24 87.92±0.22 24.71±0.48
τ= 0.7(ours) 84.91±0.21 89.37±0.18 25.81±0.42
τ= 0.9 84.12±0.26 88.41±0.25 25.02±0.47
but also yields more fluent, faithful clinical reports.
Ablation Studies.To understand the role of causality, we
ablate both its presence and the ratio used for refining (i.e.,
the percentage of the causal branch manually removed). As
shown in Table 2, removing causality causes the steepest
drop (−3.65Acc,−3.95F1,−3.34BLEU), highlighting its
central role in grounding answers in clinically meaningful
evidence. Using causality without refining partially recov-
ers performance but still introduces noisy links. Introducing
confidence-based refining steadily improves results, with the
best trade-off observed atτ= 0.7(84.91Acc,89.37F1,
25.81BLEU). Lower ratio (e.g.,τ= 0.5) allow noise to per-
sist, while higher ratio (e.g.,τ= 0.9) over-prune and reduce
recall. Causality thus drives robust reasoning by structur-
ing the search space, while manually refining calibrates the
precision–coverage trade-off by pruning unreliable links.
Table 3 shows that both re-ranking and filtering are crucial
for RAG. Removing re-ranking reduces performance (83.78
Acc, 87.20 F1, 24.61 BLEU), while removing filtering leads
to an even larger drop (82.15 / 85.40 / 23.20), indicating its
stronger role. VaryingKreveals the evidence–noise trade-
off: too few reports (K= 5) limit coverage, too many (K=
20) add noise, and the best balance is atK= 10.Table 3: Analysis of report usage in RAG with different re-
trieval settings. We report performance (mean±std over 3
runs).Re-rankingreorders retrieved reports, andFilteringre-
moves low-quality ones by score threshold.
Method Acc↑F1↑BLEU↑
RAG (Full Model,K= 10) 84.91±0.22 89.37±0.20 25.81±0.45
w/o Re-ranking 83.78±0.31 87.20±0.27 24.61±0.52
w/o Filtering 82.15±0.29 85.40±0.33 23.20±0.58
K= 5 84.10±0.25 87.40±0.28 25.00±0.49
K= 10(ours) 84.91±0.22 89.37±0.20 25.81±0.45
K= 20 84.60±0.27 88.00±0.26 25.60±0.50
4. LIMITATIONS
While MCRAG advances retrieval by incorporating causal
reasoning, several limitations remain. The framework pre-
supposes that VLM can reliably encode and expose causal
structures; however, this assumption may not hold in do-
mains characterized by highly specialized or rapidly evolving
knowledge. Moreover, the identification of causal path-
ways during inference necessitates additional model queries,
thereby increasing computational overhead and potentially
constraining scalability in practical deployments.
5. CONCLUSION
We present MCRAG, a multimodal causal retrieval frame-
work that enhances factuality and robustness in medical vi-
sion–language models. By integrating graph-based causal
reasoning within cross-modal retrieval, MCRAG achieves
state-of-the-art results on radiology-specific VQA and re-
port generation tasks. Ablation analysis further underscores
the importance of causal grounding for clinically meaningful
evidence and demonstrates the effectiveness of manual refine-
ment in improving precision. Taken together, these findings
highlight causal retrieval as a viable pathway toward safer
deployment in real-world clinical settings.

6. REFERENCES
[1] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto
Usuyama, Haotian Liu, Jianwei Yang, Tristan Nau-
mann, Hoifung Poon, and Jianfeng Gao, “Llava-
med: Training a large language-and-vision assistant for
biomedicine in one day,”Advances in Neural Infor-
mation Processing Systems, vol. 36, pp. 28541–28564,
2023.
[2] Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Yanfeng Wang,
and Weidi Xie, “Towards generalist foundation model
for radiology by leveraging web-scale 2d&3d medical
data,”arXiv preprint arXiv:2308.02463, 2023.
[3] Jiawei Chen, Dingkang Yang, Tong Wu, Yue Jiang, Xi-
aolu Hou, Mingcheng Li, Shunli Wang, Dongling Xiao,
Ke Li, and Lihua Zhang, “Detecting and evaluating
medical hallucinations in large vision language models,”
arXiv preprint arXiv:2406.10185, 2024.
[4] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu
Chen, et al., “Lora: Low-rank adaptation of large lan-
guage models.,”ICLR, vol. 1, no. 2, pp. 3, 2022.
[5] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang
Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and
Haofen Wang, “Retrieval-augmented generation for
large language models: A survey,”arXiv preprint
arXiv:2312.10997, 2023.
[6] Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong
Zhang, “Benchmarking retrieval-augmented generation
for medicine,” inFindings of the Association for Com-
putational Linguistics ACL 2024, 2024, pp. 6233–6251.
[7] Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Wei-
jia Shi, Sheng Wang, Linjun Zhang, James Zou, and
Huaxiu Yao, “Mmed-rag: Versatile multimodal rag sys-
tem for medical vision language models,”arXiv preprint
arXiv:2410.13085, 2024.
[8] Hao Chen, Hongrun Zhang, U Wang Chan, Rui Yin,
Xiaofei Wang, and Chao Li, “Domain game: Dis-
entangle anatomical feature for single domain general-
ized segmentation,” inInternational Workshop on Com-
putational Mathematics Modeling in Cancer Analysis.
Springer Nature Switzerland Cham, 2024, pp. 41–51.
[9] Nengbo Wang, Xiaotian Han, Jagdip Singh, Jing Ma,
and Vipin Chaudhary, “Causalrag: Integrating causal
graphs into retrieval-augmented generation,”arXiv
preprint arXiv:2503.19878, 2025.
[10] Weixing Chen, Yang Liu, Ce Wang, Jiarui Zhu, Guanbin
Li, Cheng-Lin Liu, and Liang Lin, “Cross-modal causalrepresentation learning for radiology report generation,”
IEEE Transactions on Image Processing, 2025.
[11] Judea Pearl,Causality: Models, Reasoning, and Infer-
ence, Cambridge university press, 2009.
[12] Jing Ma, “Causal Inference with Large Language
Model: A Survey,” Sept. 2024, arXiv:2409.09822 [cs].
[13] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon
Kim, James Glass, and Pengcheng He, “Dola: Decoding
by contrasting layers improves factuality in large lan-
guage models,”arXiv preprint arXiv:2309.03883, 2023.
[14] Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang,
Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang,
and Nenghai Yu, “Opera: Alleviating hallucination
in multi-modal large language models via over-trust
penalty and retrospection-allocation,”arXiv preprint
arXiv:2311.17911, 2023.
[15] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li,
Shijian Lu, Chunyan Miao, and Lidong Bing, “Mitigat-
ing object hallucinations in large vision-language mod-
els through visual contrastive decoding,”arXiv preprint
arXiv:2311.16922, 2023.
[16] Sunan He, Yuxiang Nie, Zhixuan Chen, Zhiyuan
Cai, Hongmei Wang, Shu Yang, and Hao Chen,
“Meddr: Diagnosis-guided bootstrapping for large-scale
medical vision-language learning,”arXiv preprint
arXiv:2404.15127, 2024.
[17] Liwen Sun, James Zhao, Megan Han, and Chenyan
Xiong, “Fact-aware multimodal retrieval augmentation
for accurate medical radiology report generation,”arXiv
preprint arXiv:2407.15268, 2024.
[18] Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li,
Gang Li, Linjun Zhang, and Huaxiu Yao, “Rule: Reli-
able multimodal rag for factuality in medical vision lan-
guage models,”arXiv preprint arXiv:2407.05131, 2024.
[19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen, “Lora: Low-rank adaptation of large lan-
guage models,”arXiv, 2021.
[20] Omid Nejati Manzari, Hamid Ahmadabadi, Hossein
Kashiani, Shahriar B Shokouhi, and Ahmad Ayatollahi,
“Medvit: a robust vision transformer for generalized
medical image classification,”Computers in biology and
medicine, vol. 157, pp. 106791, 2023.
[21] Emily Alsentzer, John R Murphy, Willie Boag, Wei-
Hung Weng, Di Jin, Tristan Naumann, and Matthew
McDermott, “Publicly available clinical bert embed-
dings,”arXiv preprint arXiv:1904.03323, 2019.

[22] Alistair EW Johnson et al., “Mimic-cxr: A large pub-
licly available database of labeled chest radiographs,”
Scientific Data, 2019.
[23] Dina Demner-Fushman, Marc D Kohli, Marc B Rosen-
man, Sonya E Shooshan, Laritza Rodriguez, Sameer
Antani, George R Thoma, and Clement J McDonald,
“Preparing a collection of radiology examinations for
distribution and retrieval,”Journal of the American
Medical Informatics Association, vol. 23, no. 2, pp.
304–310, 2016.