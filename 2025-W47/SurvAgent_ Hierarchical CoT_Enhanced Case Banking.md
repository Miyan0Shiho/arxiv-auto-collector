# SurvAgent: Hierarchical CoT-Enhanced Case Banking and Dichotomy-Based Multi-Agent System for Multimodal Survival Prediction

**Authors**: Guolin Huang, Wenting Chen, Jiaqi Yang, Xinheng Lyu, Xiaoling Luo, Sen Yang, Xiaohan Xing, Linlin Shen

**Published**: 2025-11-20 18:41:44

**PDF URL**: [https://arxiv.org/pdf/2511.16635v1](https://arxiv.org/pdf/2511.16635v1)

## Abstract
Survival analysis is critical for cancer prognosis and treatment planning, yet existing methods lack the transparency essential for clinical adoption. While recent pathology agents have demonstrated explainability in diagnostic tasks, they face three limitations for survival prediction: inability to integrate multimodal data, ineffective region-of-interest exploration, and failure to leverage experiential learning from historical cases. We introduce SurvAgent, the first hierarchical chain-of-thought (CoT)-enhanced multi-agent system for multimodal survival prediction. SurvAgent consists of two stages: (1) WSI-Gene CoT-Enhanced Case Bank Construction employs hierarchical analysis through Low-Magnification Screening, Cross-Modal Similarity-Aware Patch Mining, and Confidence-Aware Patch Mining for pathology images, while Gene-Stratified analysis processes six functional gene categories. Both generate structured reports with CoT reasoning, storing complete analytical processes for experiential learning. (2) Dichotomy-Based Multi-Expert Agent Inference retrieves similar cases via RAG and integrates multimodal reports with expert predictions through progressive interval refinement. Extensive experiments on five TCGA cohorts demonstrate SurvAgent's superority over conventional methods, proprietary MLLMs, and medical agents, establishing a new paradigm for explainable AI-driven survival prediction in precision oncology.

## Full Text


<!-- PDF content starts -->

SurvAgent: Hierarchical CoT-Enhanced Case Banking and
Dichotomy-Based Multi-Agent System for Multimodal Survival Prediction
Guolin Huang1*, Wenting Chen2*, Jiaqi Yang3, Xinheng Lyu3,
Xiaoling Luo1, Sen Yang4, Xiaohan Xing2†, Linlin Shen1†
1Shenzhen University,2Stanford University,3University of Nottingham Ningbo China,4Ant Group
Abstract
Survival analysis is critical for cancer prognosis and
treatment planning, yet existing methods lack the trans-
parency essential for clinical adoption. While recent
pathology agents have demonstrated explainability in di-
agnostic tasks, they face three limitations for survival pre-
diction: inability to integrate multimodal data, ineffective
region-of-interest exploration, and failure to leverage ex-
periential learning from historical cases. We introduce
SurvAgent, the first hierarchical chain-of-thought (CoT)-
enhanced multi-agent system for multimodal survival pre-
diction.SurvAgentconsists of two stages: (1) WSI-Gene
CoT-Enhanced Case Bank Construction employs hierarchi-
cal analysis through Low-Magnification Screening, Cross-
Modal Similarity-Aware Patch Mining, and Confidence-
Aware Patch Mining for pathology images, while Gene-
Stratified analysis processes six functional gene categories.
Both generate structured reports with CoT reasoning, stor-
ing complete analytical processes for experiential learn-
ing. (2) Dichotomy-Based Multi-Expert Agent Inference re-
trieves similar cases via RAG and integrates multimodal
reports with expert predictions through progressive inter-
val refinement. Extensive experiments on five TCGA co-
horts demonstrateSurvAgent’s superority over conven-
tional methods, proprietary MLLMs, and medical agents,
establishing a new paradigm for explainable AI-driven sur-
vival prediction in precision oncology.
1. Introduction
Survival analysis estimates patient survival time using
pathology whole slide images (WSIs) and genomic data,
providing crucial insights for cancer treatment and precision
medicine [14, 19]. While numerous studies [17, 35, 48–51]
have achieved significant performance, existing methods
*Equal contribution.
†Corresponding Author.
WSI
Gene
×2.5
LMScreen
×10
CoSaMining
WSI Attribute Checklist
 WSI Structured ReportWSI
CoT Case
Bank
Gene -based
Mathematical
Statistics
Selected Gene
Gene Report
Gene 
Knowledge Base
×20
ConfMining
Gene
CoT Case
BankHierachical WSI CoT -Enhanced Case Banking 
Gene -stratified CoT -Enhanced Case Banking 
WSI-Gene
CoT Case
Bank
 Retrieved Cases
 Multi -Expert Models
0 1 2
Time IntervalMultimodal CoT
Risk&TimeWSI-Gene CoT -Enhanced Case Bank Construction
Dichotomy -Based Multi -Expert Agent Inference 
Figure 1.SurvAgentutilizes WSI-Gene CoT-Enhanced Case
Banks through hierarchical WSI analysis and gene-stratified anal-
ysis, then performs dichotomy-based multi-expert inference by re-
trieving cases and progressively refining survival predictions.
lack transparent decision-making and interpretable explana-
tions, which are essential for clinicians to validate predic-
tions and make informed treatment decisions [1, 33]. De-
veloping explainable multimodal survival analysis methods
is therefore critical for clinical trust and adoption.
Due to the success of large language model (LLM)-based
agents, they are increasingly used in medicine for various
scenarios. These agents gather patient evidence, synthe-
size data, enable cross-specialty collaboration, and provide
transparent decision-making. Pathology agents [3, 13, 28,
32, 37, 42] are developed for WSI-based diagnostics, in-
terpretably mimicking pathologists’ reasoning through op-
erational actions (e.g., zooming, panning) while explaining
their logic. However, no agent exists specifically for sur-
vival prediction, and adapting diagnostic agents to this task
poses major challenges that limit performance.arXiv:2511.16635v1  [cs.CV]  20 Nov 2025

First, existing pathology agents [3, 13, 28, 37, 42] pri-
marily acceptsingle-modality input(i.e., WSIs), which
constrains their ability to incorporate genomic data—an es-
sential source of molecular information for understanding
tumor biology and improving survival prediction. Mor-
phological insights from histopathology and molecular pro-
files from genomics offer complementary perspectives on
tumor evolution and therapeutic response. Thus, integrat-
ing multimodal data within pathology agents is crucial for
capturing both phenotypic and molecular determinants of
prognosis, ultimately enabling more accurate and biologi-
cally informed survival prediction.
Another challenge lies in theineffective Region-of-
Interest (ROI) exploration strategiesused by current
pathology agents. Expert-based approaches [42] rely on
pathologists’ viewing behaviors to identify ROIs, but this
process is labor-intensive and yields limited training data,
hindering scalability. Automatic methods attempt to over-
come this issue but introduce new trade-offs: CPathA-
gent [37] downscales WSIs for efficiency at the cost of fine-
grained lesion details, PathFinder [13] preserves high reso-
lution but requires time-consuming sequential patch selec-
tion, and SlideSeek [3] employs tissue detection yet often
captures irrelevant or incomplete tumor regions. Overall,
existing methods either miss critical lesions, demand exces-
sive computation, or include redundant areas, highlighting
the need for an ROI mining strategy that balances accuracy,
efficiency, and coverage for robust survival prediction.
Third, current pathology agents primarily analyze each
test case in isolation,overlooking valuable prognostic ex-
periencefrom previous cases. Most existing agents [3, 13,
28, 37, 42] rely on pathology foundation models or case-
specific knowledge bases to process WSIs independently,
without leveraging information from similar patients. In
contrast, clinicians often estimate survival by referenc-
ing comparable cases, highlighting the need to incorporate
experiential knowledge. Although some general-purpose
medical agents [18, 23–25] introduce memory or database
mechanisms, they mainly store factual data while neglect-
ing the reasoning process—how clinicians integrate and
weigh evidence to reach diagnostic conclusions [7, 11, 34].
Thus, current agents need to reconstruct reasoning patterns
for each new case, underscoring the importance of integrat-
ing experiential learning with explicit reasoning pathways
for effective survival prediction.
To address these challenges, we introduceSurvAgent,
the first multi-agent system specifically designed for mul-
timodal survival prediction. Our framework consists of
two key components: (1)WSI-Gene CoT-Enhanced Case
Bank Constructionfor experiential learning, and (2)
Dichotomy-Based Multi-Expert Agent Inferencefor fi-
nal prediction.
WSI-Gene CoT-Enhanced Case Bank Constructiongenerates reasoning-based case analyses with explicit path-
ways for WSI and genomic data via two modules. Foref-
fective WSI ROI exploration, theHierarchical WSI CoT-
Enhanced Case Bankuses a multi-magnification pipeline:
(1) At low magnification,Low-Magnification Screening
(LMScreen)—PathAgent generates global WSI reports; (2)
At medium magnification,Cross-Modal Similarity-Aware
Patch Mining (CoSMining)—excludes redundant patches
by computing self-patch and self-report similarity, select-
ing patches meeting both criteria; (3) At high magnification,
Confidence-Aware Patch Mining (ConfMining)—identifies
low-confidence patches, zooms in, and applies CoSMining
to capture overlooked lesions. PathAgent extracts attributes
via a pre-defined WSI attribute checklist, generates struc-
tured reports, and creates CoT reasoning from reports and
ground-truth survival times. Toincorporate experiential
learning with explicit reasoning pathways, a self-critique
and refinement mechanism stores not only patient facts but
also the complete analytical reasoning process for survival
prediction. The WSI CoT Case Bank stores CoTs, summa-
rized reports, and survival times, enabling future cases to
benefit from both diagnostic conclusions and reasoning pro-
cesses of similar historical cases. Formultimodal data inte-
gration, theGene-Stratified CoT-Enhanced Case Bankhas
GenAgent analyze genomics by classifying genes into six
types, performing statistical analysis, and generating type-
specific reports using a knowledge base. After summariza-
tion, CoT generation, and self-critique, the Gene CoT Case
Bank stores CoTs, summarized reports, and survival times
alongside pathological data.
Dichotomy-Based Multi-Expert Agent Inference
stage leverages the constructed case banks for final pre-
diction. For test cases, the system generates hierar-
chical WSI reports through LMScreen, CoSMining, and
ConfMining, while genomic reports are produced using the
same pipeline as during case bank construction. Using
retrieval-augmented generation (RAG), similar cases with
their stored reasoning pathways are retrieved based on mul-
timodal report similarity, allowing the system to reference
both the conclusions and analytical processes from compa-
rable historical cases. An inference agent then integrates
retrieved cases, summarized reports, and predictions from
multiple expert survival models. Rather than directly pre-
dicting survival time, the agent employsdichotomy-based
reasoning: first classifying the case into coarse survival in-
tervals, then progressively refining the classification, and fi-
nally predicting exact survival time within the identified in-
terval. This hierarchical decision process, combined with
comprehensive WSI-gene reports and explicit reasoning
pathways, provides transparent and interpretable survival
predictions that align with clinical decision-making prac-
tices. Our contributions are summarized as follows:
• We proposeSurvAgent, the first multi-agent system

for multimodal survival prediction, featuring WSI-Gene
CoT-Enhanced Case Banking that stores analytical rea-
soning processes to enable experiential learning.
• We introduce Hierarchical WSI CoT-Enhanced Case
Bank with LMScreen, CoSMining, and ConfMining that
balances accuracy, efficiency, and coverage across multi-
ple magnifications.
• We propose Dichotomy-Based Multi-Expert Agent Infer-
ence that integrates retrieved cases, multimodal reports,
and expert predictions through progressive interval refine-
ment for transparent survival time prediction.
• Extensive experiments demonstrate that SurvAgent
achieves the best C-index across 5 datasets while provid-
ing interpretable multimodal reports and CoT reasoning.
2. Related Works
2.1. Multimodal Survival Prediction
Current multimodal survival analysis approaches [4, 6, 17,
35, 45, 48, 48–51] integrate pathological WSIs and ge-
nomic data to provide a more comprehensive perspective
on patient stratification and prognosis. For instance, Chen
et al.[4] employ the Kronecker product to model pairwise
feature interactions across multimodal data. Chen et al.[6]
present a multimodal co-attention transformer (MCAT)
framework that learns an interpretable, dense co-attention
mapping between WSIs and genomic features within a uni-
fied embedding space. However, they achieve strong mul-
timodal survival prediction performance at the expense of
interpretability, lacking the transparent reasoning required
for clinical trust and adoption [1, 33]. Thus, we propose
a multi-agent system that enables reliable and explainable
survival analysis through transparent decision-making.
2.2. Medical LLM-based Agent
LLM-based agents in medicine include general-purpose
systems and specialty-designed agents for radiology [26,
41], gastroenterology [39], oncology [12], and pathol-
ogy [3, 13, 28, 32, 37, 42]. The pathology agents perform
operational actions while articulating analytical logic, but
face limitations for survival prediction: they process only
WSIs without integrating genomic data [3, 13, 28, 37, 42],
their ROI exploration involves trade-offs between labor-
intensive expert annotation [42] and automatic approaches
with reduced resolution [37], excessive time [13], or irrele-
vant regions [3], and they analyze cases in isolation. While
general-purpose agents [18, 23–25] use memory mecha-
nisms, they store only factual data without reasoning pro-
cesses. Thus, we propose a dichotomy-based multi-agent
framework with cross-modal analysis, efficient patch min-
ing for ROI selection, and WSI-Gene CoT-enhanced case
banking to preserve complete prognostic reasoning.3. Method
In Fig. 2,SurvAgentcomprises two stages: WSI-Gene
CoT-Enhanced Case Bank Construction and Dichotomy-
Based Multi-Expert Agent Inference. In the first stage, we
build case banks with pathological and genomic reason-
ing pathways. For WSI analysis, a hierarchical pipeline
operates at increasing magnifications: LMScreen gener-
ates global reports at×2.5, CoSMining excludes redun-
dant patches while retaining important regions at×10via
self-patch and self-report similarity, and ConfMining iden-
tifies low-confidence patches at×20for further exploration.
PathAgent extracts attributes using a pre-defined check-
list, generates structured reports, and creates CoT reason-
ing with self-critique. CoTs, reports, and survival times
are stored in the WSI CoT Case Bank. For genomic data,
GenAgent performs gene-stratified analysis and generates
type-specific reports, stored in the Gene CoT Case Bank.
During inference, test cases undergo hierarchical analysis
to generate multimodal reports. Using RAG, similar cases
with stored CoTs are retrieved. Inference agent integrates
retrieved cases, multimodal reports, and expert predictions
through dichotomy-based reasoning, progressively refining
survival intervals to predict exact survival time.
3.1. WSI-Gene CoT-Enhanced Case Bank Con-
struction
To enable experiential learning with explicit reasoning, we
construct a WSI-Gene CoT-Enhanced Case Bank that stores
both patient facts and reasoning traces, mimicking how clin-
icians draw on experience from similar cases. It comprises
two components: (1) Hierarchical WSI CoT-Enhanced Case
Bank performs multi-magnification analysis through LM-
Screen, CoSMining, and ConfMining to generate patholog-
ical reports and reasoning pathways, and (2) Gene-Stratified
CoT-Enhanced Case Bank conducts systematic analysis
across six gene types to produce genomic reports and rea-
soning pathways. Both components generate summarized
reports, create CoT reasoning based on ground-truth sur-
vival times, apply self-critique for refinement, and store
triplets into case banks for retrieval during inference.
3.1.1. Hierachical WSI CoT-Enhanced Case Bank
Low-Magnification Screening (LMScreen).To obtain
comprehensive global understanding of the WSI while
maintaining computational efficiency, we perform initial
analysis at×2.5magnification. Given a WSIW, we first
downsample it to obtain the low-magnification representa-
tionW 2.5. A PathAgentA wsiprocesses this representation
to generate a global report:R global =A wsi(W2.5). This
global reportR global captures overall tissue architecture.
Cross-Modal Similarity-Aware Patch Mining (CoSMin-
ing).For effective WSI ROI exploration, we introduce CoS-
Mining to mine fine-grained high-magnification patches by

WSI x 2.5 factor×2.5 report
PathAgent
Observation area 1
Observation area 2×2.5
×10
×20
WSI Preprocessing
Background Removal
Patch-Level Spatial Clustering
Patch-Level Tumor classification
WSI Image
Gene DataPathAgent
GenAgent
Selected Gene
ACP6_cnvBCL9_cnvCCDC120_rnaseqDRD1_rnaseq+02∞Survival time interval (years)
Gene-basedMathematicalStatistics
WSI-Gene CoT CaseBankSummarized report
Summarizedreport
Dichotomy-Based Multi-Expert Agent Inference
Retrieved Cases
Depth of InvasionWSI ReportVariant HistologyMargin Status…Comparison with retrieved casesGene ReportMathematicalStatistics
Selected GeneGenerated CoT
Final Result**Risk**: -3.235**Survival Time**: 20.48 monthsPatient reportSummarized report
Survival time interval WSI
Inference-Agent
PathAgent
Confidence on Visual Evidence & Report 
❌
PathAgent
PatchConfidence on Visual Evidence & Report 
✅Confidence-Aware ReflectionFurtherAnalysis×20 report
×10 report
Confidence-Aware Patch Mining (ConfMining) 
CMSPM
……ReportSelf-PatchSimilarity Matrix
WSI x 10 factorPathAgent
Cross-Modal Similarity-Aware Patch Mining(CoSMining) 
Similarity Matrix
patch report
Selected Patches !
Selected Patches 
…
∩……×10 report
Gene KnowledgebaseGene-stratified CoT-Enhanced Case Bank ConstructionReport for each gene type
Gene Data
GenAgent
Selected gene
ACP6_cnvBCL9_cnvCCDC120_rnaseqDRD1_rnaseq
GenAgentGene-basedMathematicalStatistics
Statistics for Gene Type 1…
Statistics for Gene Type 2
Statistics for Gene Type 5
Statistics for Gene Type 6Summarized reportGenerated CoTGT Survival Class / TimeGene-Only
CoT CaseBankSummarized report
GT Survival Class & TimeGenerated CoT
GenAgent
…
Tumor Suppressor Genes
Oncogenes
Protein Kinases
Cell Differentiation Markers
...
Gen CoT CaseBank
Self-Critique
Web
Knowledge BaseSearchAgent
ExpertReview
WSI AttributeChecklist
WSI AttributeChecklistSummarized report
Summarized ReportGenerated CoTGT Survival Class / TimeWSI-Only
Generated CoTWSI CoT CaseBankSummarizeSelf-critiquePathAgent
WSI Attribute ChecklistPathAgent
Hierachical WSI CoT-Enhanced Case Bank Construction
Structured Report
×2.5 report
×10 report
×20 report
GT Survival Class & TimeWSI Attribute Checklist Generation
Low-Magnification Screening (LMScreen)
LMScreen
CoSMining
ConfMining
×2.5
×10
×20
Survival Time
Inference-Agent
Multi-Expert Models
×20Zoom in
PathAgent
……Compute Intersection between " and #Selected Patches $"#LMScreen
×2.5
CoSMining
×10
ConfMining
×20
×2.5 report
×10 report
×20 report
Initial Report×2.5 report
×10 report
×20 report
×2.5
×10
×20
Structured Report
×2.5
LMScreen
×10
CoSMining
×20
ConfMining
×2.5
LMScreen
×10
CoSMining
×20
ConfMining
RetrieveRetrieveRAGDiscussion
Generated CoT+
Retrieved Cases012Survival time interval (years)
0123∞Survival time interval (years)ExcludeSimilarPatchesExcludeSimilarPatches
Figure 2. Overview ofSurvAgent. (1)WSI-Gene CoT-Enhanced Case Bank ConstructionincludesHierarchical WSI CoT-Enhanced
Case Bankthat progressively analyzes WSIs at multiple magnifications through LMScreen, CoSMining, and ConfMining, andGene-
Stratified CoT-Enhanced Case Bankfor gene statistical analysis. PathAgent and GenAgent generate structured reports and CoT reasoning
with self-critique for their respective case banks. (2)Dichotomy-Based Multi-Expert Agent Inferenceuses RAG for retrieval and
integrates retrieved cases, reports, and expert predictions for progressive survival time prediction from coarse to fine-grained intervals.
select diagnostically important patches while excluding re-
dundant regions through self-patch and self-report similar-
ities. Specifically, given the downsampled WSIW 10at
×10magnification, we partition it intoNnon-overlapping
patches{p i}N
i=1. To eliminate visual redundancy, we con-
struct a self-patch similarity matrixSv∈RN×N, where
each element is computed as:Sv
ij=sim(ϕ(p i), ϕ(p j)),
whereϕ(·)denotes a pathology foundation model en-
coder [43] and sim(·,·)represents cosine similarity. We
identify and remove patches with high visual similarity by
selecting patches whose maximum similarity to others ex-
ceeds thresholdτ v:Pv
selected ={p i|max j̸=iSv
ij< τv}.
To ensure semantic diversity, PathAgent generates pre-
liminary reports{R i}N
i=1for all patches. We then construct
a self-report similarity matrixSt∈RN×Nin the textualsemantic space:St
ij=sim(ψ(R i), ψ(R j)),whereψ(·)
represents text embedding encoded by a text encoder [10].
Similarly, we select semantically diverse patches through
the thresholdτ t:Pt
selected ={p i|max j̸=iSt
ij< τt}.
The final selected patches are obtained through inter-
section of both criteria:P 10=Pv
selected∩ Pt
selected ,en-
suring patches are both visually distinctive and semanti-
cally informative. PathAgent then generates detailed reports
{R10
k}|P10|
k=1for these selected patches.
Confidence-Aware Patch Mining (ConfMining).Af-
ter CoSMining identifies potentially informative patches at
10×magnification, not all patches require further high-
magnification analysis. To efficiently allocate compu-
tational resources, ConfMining introduces a Confidence-
Aware Reflection mechanism that selectively determines

which patches warrant deeper examination at20×magnifi-
cation based on PathAgent’s analytical confidence. For each
patchp k∈ P 10with its corresponding reportR10
k, PathA-
gent predicts the confidence level of the reports from three
categories: low, medium, and high. When PathAgent as-
signs a low confidence level—indicating uncertainty about
morphological features, ambiguous cellular patterns, or the
need for finer details—the patch is selected for hierarchical
20×magnification analysis.
For each low-confidence patchp k∈ P low-conf , we zoom
in to×20magnification to obtain its high-resolution version
p20
kand partition it intoMsub-patches{p20
k,m}M
m=1. To
avoid exhaustive analysis of all sub-patches, we apply the
CoSMining strategy at this finer scale to extract the most
informative sub-patches through cross-modal similarity fil-
tering:P20
k=CoSMining(p20
k;τv, τt).The final set of
high-magnification patches isP 20=S
kP20
k, and PathA-
gent generates corresponding detailed reports{R20
m}|P20|
m=1
for these selected sub-patches. This two-stage confidence-
driven and similarity-aware mining process ensures thor-
ough analysis of uncertain regions while maintaining com-
putational efficiency by focusing only on the most relevant
fine-grained features within low-confidence patches.
WSI Attribute Checklist Generation.To ensure
structured and clinically relevant analysis, we em-
ploy a search agentA search to establish a WSI at-
tribute checklistC WSI. The search agent queries med-
ical knowledge basesK med [28] and online resources
Dweb[30] to identify prognostically important attributes:
Craw=A search(Kmed,Dweb;qsurvival ), whereq survival repre-
sents survival-prediction-specific queries. The raw check-
listC rawis then reviewed and refined by clinical experts to
obtainC WSI={a 1, a2, . . . , a K}, where eacha krepresents
a key pathological attribute (e.g., tumor grade, necrosis ex-
tent, lymphocytic infiltration). We include 16 key attributes.
Using this checklist, PathAgent extracts structured infor-
mation from all hierarchical reports:
Rstruct=A wsi({R global,{R10
k},{R20
m}};C WSI),
and generates a summarized reportRWSI
sumby removing re-
dundant information. Finally, given the summarized report
RWSI
sumand GT survival timet GT, PathAgent generates CoT
reasoning: CoT WSI=A wsi(RWSI
sum, tGT).
To ensure the correctness of the CoT, we utilize a self-
critique mechanism to evaluate the generated CoT. Specif-
ically, we employ Qwen2.5-32B as the quality validation
functionV(·)to assess the CoT and generate both a qual-
ity level (low or high) and detailed critique. The refinement
process is formulated as:
CoTrefined
WSI =(
CoT WSI ifV(CoT WSI) =high
Awsi(CoT WSI,Critique)Otherwise,
(1)whereV(CoT WSI)returns the quality level and generates
a critique, andA wsi(·,·)denotes PathAgent’s refinement
operation that revises the CoT based on critique feed-
back until high quality is achieved. The final triplet
(RWSI
sum,CoTrefined
WSI, tGT)is stored in the WSI CoT Case Bank
BWSI. Through this hierarchical pipeline, from global
screening to cross-modal mining and confidence-aware re-
finement, we construct a comprehensive WSI case bank
capturing multi-scale morphological features with explicit
reasoning pathways for experiential survival prediction.
3.1.2. Gene-Stratified CoT-Enhanced Case Bank
For multimodal data integration, we introduce a Gene-
Stratified CoT-Enhanced Case Bank to systematically an-
alyze genomic data. Since raw genomic data is highly
abstract with individual genes often lacking direct clinical
value, we classify genes by functional roles into six prog-
nostically important categories: Tumor Suppressor Genes,
Oncogenes, Protein Kinases, Cell Differentiation Markers,
Transcription Factors, and Cytokines and Growth Factors.
Given genomic dataG, letG ldenote the gene subset of
typelwherel={1, . . . , L},(L= 6). For each type,
we compute statistical featuress l= [µ l, ml, rl
mut], where
µl,ml, andrl
mutrepresent mean expression, median, and
mutation ratio, providing comprehensive quantitative char-
acterization of expression and mutation patterns.
These statistics are submitted to GenAgentA genfor pre-
liminary analysis. GenAgent analyzes the statistical in-
formation with gene knowledge baseK gene [44] to au-
tonomously select genes with significant prognostic impact,
Gl=Aselect
gen(sl,Gl;Kgene),whereG∗
l⊂ G lcontains se-
lected important genes. GenAgent then retrieves raw ex-
pression data and consultsK geneto understand each gene’s
implications. Through coarse-to-fine analysis from statis-
tics to gene details, GenAgent produces type-specific re-
ports,Rgene
l=Areport
gen(sl,G∗l;K gene).
After analyzing all six categories, GenAgent generates a
comprehensive genomic reportRgene
sum. Similar to WSI anal-
ysis, GenAgent generates CoT reasoning from the genomic
report and GT patient survival class and time: CoT gene=
Agen(Rgene
sum, tGT), followed by self-critique and refinement:
CoTrefined
gene =(
CoT gene ifV(CoT gene) =high
Agen(CoT gene,Critique)Otherwise.
(2)
The triplet(Rgene
sum,CoTrefined
gene, tGT)is stored in Gene CoT
Case BankB genefor inference-time retrieval. Through this
gene-stratified pipeline, from statistical characterization to
targeted gene selection and CoT-enhanced analysis, we con-
struct a comprehensive genomic case bank capturing func-
tional genomic patterns with explicit reasoning pathways
for knowledge-guided survival prediction

Table 1. Comparison of survival prediction performance (C-index) across different models and modalities on five TCGA cancer cohorts. G:
Genomic modality, H: Histopathology modality. “*” indicate best results from our reimplementation; Others are from original publications.
Model Modality BLCA BRCA GBMLGG LUAD UCEC Overall
(N = 373) (N = 956) (N = 480) (N = 569) (N = 453)
Conventional Methods
SNN* [21] G 0.541±0.016 0.466±0.058 0.598±0.054 0.539±0.069 0.493±0.096 0.527
SNNTrans [50] G 0.646±0.043 0.648±0.058 0.828±0.016 0.634±0.049 0.632±0.032 0.678
AttnMIL [16, 50] H 0.605±0.045 0.551±0.077 0.816±0.011 0.563±0.050 0.614±0.052 0.630
MaxMIL [50] H 0.551±0.032 0.597±0.055 0.714±0.057 0.596±0.060 0.563±0.055 0.604
DeepAttnMISL [45, 46] H 0.504±0.042 0.524±0.043 0.734±0.029 0.548±0.050 0.597±0.059 0.581
M3IF [22, 50] G+H 0.636±0.020 0.620±0.071 0.824±0.017 0.630±0.031 0.667±0.029 0.675
HFBSurv [50] G+H 0.640±0.028 0.647±0.035 0.838±0.013 0.650±0.050 0.642±0.045 0.683
MOTCat* [45] G+H 0.674±0.024 0.684±0.011 0.831±0.028 0.674±0.036 0.667±0.051 0.706
MCAT* [5] G+H 0.645±0.053 0.601±0.0690.852±0.0280.636±0.043 0.634±0.018 0.674
CCL* [50] G+H 0.652±0.034 0.593±0.058 0.845±0.012 0.640±0.059 0.668±0.041 0.680
Proprietary MLLMs
Gemini-2.5-Pro* [8] G+H 0.572±0.031 0.555±0.055 0.551±0.026 0.531±0.062 0.498±0.067 0.541
Claude-4.5* [2] G+H 0.545±0.027 0.555±0.046 0.505±0.053 0.509±0.059 0.479±0.034 0.519
GPT-5* [31] G+H 0.576±0.038 0.434±0.057 0.493±0.053 0.510±0.087 0.495±0.083 0.502
Medical Agents
MDAgent* [20] G+H 0.558±0.040 0.482±0.064 0.495±0.040 0.524±0.064 0.509±0.049 0.514
MedAgent* [38] G+H 0.515±0.039 0.510±0.031 0.483±0.020 0.485±0.050 0.551±0.066 0.509
SurvAgent (Ours)G+H0.683±0.022 0.695±0.0130.833±0.0290.676 ± 0.036 0.676±0.052 0.713
3.2. Dichotomy-Based Multi-Expert Agent Infer-
ence Stage
For a test case with WSIW testand genomic dataG test, we
first generate multimodal reports using the same hierarchi-
cal pipeline. For WSI analysis, we apply the hierarchi-
cal analysis pipeline including LMScreen, CoSMining, and
ConfMining to extract hierarchal WSI reportsRWSI
test. For
genomic data, we perform gene-stratified analysis across six
gene types to characterize genomic reportsRgene
test.
Using retrieval-augmented generation (RAG), we re-
trieveKsimilar cases from both case banks based on mul-
timodal report similarity:
Bretrieved =RAG(RWSI
test,Rgene
test;BWSI,Bgene, K),(3)
where each retrieved case contains its summarized reports,
CoT reasoning, and survival class and time.
Additionally, we collect predictions fromMex-
pert survival models [6, 45, 50]:{ ˆtm}M
m=1 =
{Mm(W test,Gtest)}M
m=1, whereM mrepresents them-
th expert model. The inference agentA infer performs
dichotomy-based reasoning by progressively refining sur-
vival intervals. We define a hierarchical interval structure
withDdichotomy levels. At the first level, the agent per-
forms coarse classification:
y1=A infer(Bretrieved ,RWSI
test,Rgene
test,{ˆtm};level= 1),(4)
wherey 1∈ {1,2}divides cases into two broad survival
categories. At each subsequent leveld={2, . . . , D}, the
agent further refines the classification within the selectedinterval:
yd=A infer(Bretrieved ,RWSI
test,Rgene
test,{ˆtm}, yd−1;level=d),
(5)
progressively narrowing the survival interval until reaching
the finest granularity. Finally, the exact survival time is pre-
dicted within the identified interval:
ˆtfinal=A infer(Bretrieved ,RWSI
test,Rgene
test,{ˆtm}, yD).(6)
The inference agent outputs comprehensive results includ-
ing the predicted survival time ˆtfinal, multimodal reports
(RWSI
test,Rgene
test), and an inference reasoning reportR reasoning
that documents the complete decision-making process, pro-
viding transparency for clinical validation.
4. Experiments
4.1. Datasets and Settings
Datasets and Evaluation.To ensure fair comparison,
we follow prior protocols using five-fold cross-validation
on five datasets: Bladder Urothelial Carcinoma (BLCA),
Breast Invasive Carcinoma (BRCA), Glioblastoma and
Lower Grade Glioma (GBMLGG), Lung Adenocarci-
noma (LUAD), and Uterine Corpus Endometrial Carcinoma
(UCEC). Data volumes remain consistent across datasets
(details in Table 1). We evaluate using the Concordance In-
dex (C-index)[15], Kaplan-Meier survival curves[19], and
Log-rank test [29] to assess survival differences between
risk groups and validate prediction reliability.
Implementation.For each WSI, tissue regions are seg-
mented via Otsu’s thresholding. Non-overlapping256×256

Table 2. Ablation study of different components inSurvAgentframework.
WSI CoT bank Gene CoT bank Inference BLCA BRCA GBMLGG LUAD UCEC Overall
0.452±0.030 0.421±0.033 0.463±0.024 0.498±0.086 0.470±0.045 0.461
✓0.612±0.053 0.542±0.095 0.791±0.024 0.559±0.039 0.585±0.050 0.618
✓0.539±0.016 0.455±0.056 0.591±0.050 0.545±0.073 0.481±0.082 0.522
✓0.664±0.041 0.665±0.012 0.813±0.024 0.650±0.039 0.652±0.052 0.689
✓ ✓ ✓0.683±0.022 0.695±0.013 0.833±0.029 0.676±0.033 0.676±0.052 0.713
0 50 100 150
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 0.0690BLCA (TCGA)
Low Risk
High Risk
0 50 100 150 200 250 300
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 0.9397BRCA (TCGA)
Low Risk
High Risk
0 50 100 150 200
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 0.0131GBMLGG (TCGA)
Low Risk
High Risk
0 50 100 150 200 250
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 0.1917LUAD (TCGA)
Low Risk
High Risk
0 50 100 150 200
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 0.9488UCEC (TCGA)
Low Risk
High Risk
0 50 100 150
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 0.0164BLCA (TCGA)
Low Risk
High Risk
0 50 100 150 200 250 300
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 0.8923BRCA (TCGA)
Low Risk
High Risk
0 50 100 150 200
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 0.4970GBMLGG (TCGA)
Low Risk
High Risk
0 50 100 150 200 250
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 0.9221LUAD (TCGA)
Low Risk
High Risk
0 50 100 150 200
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 0.6400UCEC (TCGA)
Low Risk
High Risk
0 50 100 150
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 1.09e-05BLCA (TCGA)
Low Risk
High Risk
0 50 100 150 200 250 300
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 1.83e-06BRCA (TCGA)
Low Risk
High Risk
0 50 100 150 200
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 1.77e-24GBMLGG (TCGA)
Low Risk
High Risk
0 50 100 150 200 250
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 7.53e-06LUAD (TCGA)
Low Risk
High Risk
0 50 100 150 200
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 6.85e-04UCEC (TCGA)
Low Risk
High Risk
0 50 100 150
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 1.84e-06BLCA (TCGA)
Low Risk
High Risk
0 50 100 150 200 250 300
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 6.65e-06BRCA (TCGA)
Low Risk
High Risk
0 50 100 150 200
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 1.24e-27GBMLGG (TCGA)
Low Risk
High Risk
0 50 100 150 200 250
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 3.34e-07LUAD (TCGA)
Low Risk
High Risk
0 50 100 150 200
Time (Months)0.00.20.40.60.81.0Overall Survivalp-value = 2.71e-05UCEC (TCGA)
Low Risk
High RiskGemini-2.5-pro MDAgent MOTCat SurvAgent
Figure 3. Kaplan-Meier Analysis of predicted high-risk (red) and low-risk (blue) groups on five cancer datasets and their p-values. Shaded
areas refer to the confidence intervals.
patches are extracted from tissue areas at20×magnification
and processed with CLAM [27] to extract features for sur-
vival prediction models. Genomic data uses an SNN [21]
encoder. CoSMining parameters:τ v=τt= 0.93; RAG:
K= 3; inference:D= 2. Code will be publicly released.
4.2. Comparison with State-of-the-Arts
To prove the superiority of ourSurvAgent, we compare it
against SOTA approaches across five TCGA cancer cohorts
in survival prediction, including conventional methods (uni-
modal and multimodal), leading proprietary MLLMs (Gem-
ini 2.5 Pro [8], Claude 4.5 [2], GPT-5 [31]), and medical
multi-agent systems (MedAgent [38] and MDAgent [20]),
as shown in Table 1.
Comparison with Conventional Methods.OurSur-
vAgentdemonstrates superior performance over conven-
tional survival prediction methods across most cohorts.
Compared to the best conventional baseline MOTCat,
SurvAgentachieves improvements of 0.9%, 1.1%, and
0.2% in C-index on BLCA, BRCA, and LUAD, respec-tively, with 0.7% overall gain. Compared to unimodal
approaches,SurvAgentsignificantly surpasses genomic-
based (SNNTrans: 0.678) and histopathology-based meth-
ods (AttnMIL: 0.630). The key advantage lies in our hier-
archical WSI analysis and CoT-enhanced case banking en-
abling experiential learning, unlike conventional feature fu-
sion methods, leading to more robust predictions.
Comparison with Proprietary MLLMs.General-purpose
proprietary MLLMs exhibit poor performance on sur-
vival prediction despite advanced capabilities elsewhere.
OurSurvAgentsubstantially outperforms Gemini-2.5-Pro,
Claude-4.5, and GPT-5 by 17.2%, 19.4%, and 21.1% in
overall C-index. On GBMLGG, the gap is most pro-
nounced:SurvAgentachieves 0.833 versus Gemini-2.5-
Pro (0.551), Claude-4.5 (0.505), and GPT-5 (0.493). Even
where proprietary MLLMs perform relatively better, such
as Gemini-2.5-Pro on BLCA (0.572),SurvAgentachieves
11.1% gains. These results show that general MLLMs
struggle with medical prediction tasks requiring domain

Initial WSITCGA -XF-A9SU
×10
Zoom In
Summary All Gene Report
The patient's genomic profile reveals marked
instability, with key alterations intumor
suppressors (TP53,CDKN 2A,PTEN) showing
deletion ordownregulation, and oncogenes
(ERBB 2,FGFR 3,MYC) amplified and
upregulated, driving proliferation .Kinases
AURKA and BRAF are elevated, while MET is
reduced, disrupting signaling pathways .
Differentiation markers (GATA 3,TP63,KRT5)
are downregulated, indicating poor
differentiation, whereas CDH 1isunusually
elevated .Transcription factors E2F3and MYC
are overexpressed, alongside functional loss
ofRB1and TP53.……
WSI Attribute ChecklistTumor grade: High -grade
Depth of Invasion: Deep muscle invasion
Lymphovascular Invasion: Present
Perineural Invasion: Present
Lymph Node Metastasis: 1-2 nodes
Margin Status: Negative
Tumor Morphology: ……
Carcinoma in Situ: Absent
Variant Histology: Lymphoepithelioma -like
Squamous Differentiation: Focal
Glandular Differentiation: Extensive
Micropapillary Component: Unknown
Plasmacytoid Component: Unknown
Sarcomatoid Differentiation: Focal
Lymphocytic Infiltration: Mild
Necrosis Percentage: 10-30%
Summary: High -grade malignant neoplasm, ……
Summarized Report
Risk Level: High (0 -12 months)
Key Evidence: 
1. **Depth of Invasion**: 
The tumor has invaded the 
muscularis propria, 
indicating a more advanced 
stage of bladder cancer.
2. **Perineural Invasion**: 
The presence of perineural 
invasion suggests aggressive 
tumor behavior and is 
associated with a poorer 
prognosis……GT Survival Class & Time
Risk Level: High  ( 0 -12 months) 
Survival Time: 5.16 monthsWSI-Gene
CoT Case
Bank
WSI Report
WSI-Gene
CoT Case
Bank
Retrieved Cases
RAG
Inference -Agent+
Expert Model
:low
:High
:low
0 1 2 3 ∞Survival 
Time Interval (Years)Classification & ClusteringObservation 
Area
CoT **Reason**: 
The current patient's genomic report indicates …genes (CDKN 2A
and RB1)and upregulation ofoncogenes (ERBB 2),protein kinases
(AURKA and MET), and cytokines and growth factors .…The WSI
report highlights alow-grade neoplasm with glandular
differentiation and sarcomatoid differentiation, …However, the
unknown status ofcritical factors such asdepth ofinvasion, ...
Comparing with the historical cases, the presence ofsarcomatoid
differentiation and significant genomic instability aligns closely
with case 1,which had ahigh -intermediate risk level and asurvival
time of21.85months .However, the authoritative pathologists
have provided asurvival time range of0-12months forthis case …
**Risk** :-2.001
**Survival Time** :6.25monthsFinal Answer
The image displays asection ofbladder tissue with
distinct morphological features .Inthe upper left
quadrant, there isadense, irregular arrangement of
c e l l s , s u g g e s t i n g ap o s s i b l e n e o p l a s t i c
process .…adenocarcinoma .…structures isfibrous
and appears desmoplastic ,indicating areactive
process often seen inmalignant tumors .…the
findings areconsistent with adenocarcinoma ofthe
bladder .
×2.5 Report
×20 Report
The image displays asection oftissue with a
dense cellular arrangement .The cells are
predominantly spindle -shaped, with elongated
nuclei, and are dispersed throughout the field .
Mitotic figures are present, suggesting active cell
division .The overall architecture and cellular
morphology are consistent with aspindle cell
neoplasm .Given thecontext ofthebladder, these
features could be indicative ofasarcomatoid
carcinoma oraspindle cell variant ofurothelial
carcinoma .
×20 Report
The image displays asection oftissue with a
dense cellular arrangement .The cells are
predominantly spindle -shaped, with elongated
nuclei, and are dispersed throughout the field .
Mitotic figures are present, suggesting active cell
division .The overall architecture and cellular
morphology are consistent with aspindle cell
neoplasm .Given thecontext ofthebladder, these
features could be indicative ofasarcomatoid
carcinoma oraspindle cell variant ofurothelial
carcinoma .
×10 Report
…adense infiltration ofsmall, round, darkly stained
cells,…The central region shows amore fibrous
stroma with scattered spindle -shaped cells,…there are
large, irregularly shaped, pale-staining areas, which
may represent edema orcystic changes .…
suggests achronic inflammatory process, possibly
chronic cystitis orareactive change .However,
without additional clinical context orhigher
magnification, adefinitive diagnosis isnot
possible .
×10 Report
…adense infiltration ofsmall, round, darkly stained
cells,…The central region shows amore fibrous
stroma with scattered spindle -shaped cells,…there are
large, irregularly shaped, pale-staining areas, which
may represent edema orcystic changes .…
suggests achronic inflammatory process, possibly
chronic cystitis orareactive change .However,
without additional clinical context orhigher
magnification, adefinitive diagnosis isnot
possible .
Tumor Suppressor Genes
Oncogenes
Protein Kinases
Cell Differentiation Markers
Transcription Factors
Cytokines and Growth Factors
Gene Type
Based on an overall situation 
analysis, ineed to conduct a 
detailed analysis of the 
following specific gene:
- **TP53**
- **RB1**
- **CDKN2A** ……
GeneAgent Search 
Specific Gene
Gene Knowledge Base
**TP 53**:This gene isa
critical tumor suppressor
and isoften involved in
bladder cancer .Its CNV
status and expression levels
should beclosely examined .
**RB 1**:This gene is……
GenAgent
Report for Each Gene Type
Gene Copy Number Variation (CNV) Statistics:
Mean:0.4706
Median:1.0000
Variation Proportion:
-Proportion of genes with amplification:0.5294
-Proportion of genes with deletion:0.0588
……Gene -based 
Mathematical Statistics
Summary of  Each Gene Type
The patient's Tumor Suppressor 
Genes cohort demonstrates a 
slight overall gain in copy number 
with 30% of genes showing 
amplification and 6.67% showing 
deletion. RNA -seq analysis 
indicates overall upregulation, 
with 54.55% of genes upregulated 
and 45.45% downregulated. 
Notably, TP53 shows 
amplification, potentially leading 
to loss of heterozygosity and 
inactivation. ……
PathAgent
Gene Report
Dichotomy -Based Multi -Expert 
Agent Inference
×2.5
LMScreen
×10
CoSMining
×20
ConfMining
PathAgent
Figure 4. Explainability analysis through visualization on case TCGA-XF-A9SU
knowledge and structured multimodal reasoning, highlight-
ing the need for specialized models likeSurvAgent.
Comparison with Medical Multi-Agent Systems.Our
SurvAgentsignificantly outperforms existing medical
multi-agent systems, achieving 19.9% and 20.4% overall C-
index improvements over MDAgent and MedAgent. Across
all cohorts,SurvAgentconsistently demonstrates superior
performance: improvements range from 12.5% (BLCA) to
33.8% (GBMLGG) over MDAgent, and 16.8% (BLCA) to
35.0% (GBMLGG) over MedAgent. These substantial gaps
highlight advantages of our task-specific design: modality-
specific agents, CoT-enhanced case banking for experiential
learning, and dichotomy-based inference for transparent de-
cisions, demonstrating that domain-specific multi-agent ar-
chitectures significantly outperform general-purpose medi-
cal agents in complex clinical prediction.
4.3. Ablation Study
To evaluate the effectiveness of each proposed component,
we conduct ablation studies by systematically removing in-
dividual modules. In Table 2,SurvAgentcomprises three
main components: the Hierarchical WSI CoT-Enhanced
Case Bank, the Gene-Stratified CoT-Enhanced Case Bank,
and the Dichotomy-Based Multi-Expert Agent Inference
stage. When integrating only the WSI CoT bank with base-
line survival models, the overall C-index improves from
0.461 to 0.618, showing the substantial value of hierarchi-
cal pathological case-based reasoning. Similarly, incorpo-
rating only the Gene CoT bank yields an overall C-index of0.522, confirming that genomic experiential learning also
contributes to performance improvement. When incorpo-
rating the proposed inference pipeline, the performance of
baseline model boosts by 0.22 in C-index, suggesting its ef-
fectiveness. The completeSurvAgentframework, which
integrates both case banks with dichotomy-based multi-
agent inference, achieves the highest overall C-index of
0.713, demonstrating the synergistic benefits of multimodal
case-based reasoning and progressive interval refinement.
4.4. Patient Stratification
Beyond C-index, patient stratification into distinct risk sub-
groups is critical for cancer prognosis. We compared Sur-
vAgent with top models from each category: MOTCat
(conventional), Gemini-2.5-pro (MLLM), and MDAgent
(multi-agent) using Kaplan-Meier curves (Fig. 3). MLLMs
and multi-agent frameworks failed to discriminate high-
risk from low-risk populations effectively. Gemini-2.5-pro
and MDAgent showed non-significant results (p>0.05) on
80% (4/5) of datasets, indicating unstable feature-outcome
associations. SurvAgent achieved 100% statistical signif-
icance (all p<0.05). On GBMLGG, SurvAgent demon-
strated exceptional stratification (p=1.24e-27). Compared
to MOTCat, SurvAgent achieved comparable or better p-
values, demonstrating its ability to leverage WSI and ge-
nomic features for robust survival prediction. This per-
formance derives from our WSI-Gene CoT-Enhanced Case
Bank and Dichotomy-Based Multi-Agent Inference, which
together enable interpretable, accurate, and progressively

refined risk stratification.
4.5. Explainability Analysis
To demonstrate the explainability and accuracy ofSur-
vAgent, we present a comprehensive case study in the
Fig.1 for case TCGA-XF-A9SU. The hierarchical WSI
analysis begins with LMScreen identifying adenocarcinoma
with “dense, irregular arrangement of cells” and desmo-
plastic structures. CoSMining’s ×10 report reveals ”dense
infiltration of small, round, darkly stained cells” with fi-
brous stroma and spindle-shaped cells, noting “without ad-
ditional clinical context or higher magnification, a definitive
diagnosis is not possible.” ConfMining’s ×20 report then
identifies critical features: “predominantly spindle-shaped
cells with elongated nuclei” and “mitotic figures present,”
with morphology “consistent with spindle cell neoplasm
indicative of sarcomatoid carcinoma“—prompting higher
magnification analysis. The WSI Attribute Checklist ex-
tracts 15 features including deep muscle invasion, perineu-
ral invasion, and focal sarcomatoid differentiation, summa-
rizing as “high-grade malignant neoplasm with perineural
invasion and necrosis.” GenAgent analyzes genes (TP53,
RB1, CDKN2A), revealing “TP53 amplification poten-
tially leading to loss of heterozygosity,” “marked instabil-
ity with tumor suppressors (TP53, CDKN2A, PTEN) dele-
tion/downregulation, oncogenes (ERBB2, FGFR3, MYC)
amplified/upregulated,” and “overexpressed E2F3/MYC
with functional loss of RB1/TP53.” The CoT reasoning doc-
uments muscularis propria invasion and perineural inva-
sion as aggressive indicators, classifying as high-risk (0-
12 months) the same as the GT low-risk labeling (5.16
months). The final inference agent reconciles conflicting
signals—e.g., “sarcomatoid differentiation and genomic in-
stability align with case1 (21.85 months), but experts esti-
mate 0–12 months”—and through dichotomy-based reason-
ing predicts 6.25 months (risk = –2.001) versus the ground
truth 5.16 months, transparently balancing contradictory ev-
idence for interpretable clinical reasoning.
5. Conclusion
We presentSurvAgent, the first multi-agent system for
multimodal survival prediction. Through WSI-Gene CoT-
Enhanced Case Banking, we enable experiential learning by
storing analytical reasoning processes from historical cases.
Our hierarchical pipeline balances accuracy, efficiency, and
coverage via LMScreen, CoSMining, and ConfMining,
while dichotomy-based inference ensures transparent pre-
dictions. Extensive experiments show our superiority to
conventional methods, MLLMs, and medical agents with
consistent statistical significance in patient stratification.References
[1] Qaiser Abbas, Woonyoung Jeong, and Seung Won Lee. Ex-
plainable ai in clinical decision support systems: A meta-
analysis of methods, applications, and usability challenges.
InHealthcare, page 2154. MDPI, 2025. 1, 3
[2] Anthropic. Claude 4.5 [large language model], 2025. Ac-
cessed 2025-11-14. 6, 7
[3] Chengkuan Chen, Luca L Weishaupt, Drew FK Williamson,
Richard J Chen, et al. Evidence-based diagnostic reasoning
with multi-agent copilot for human pathology.arXiv preprint
arXiv:2506.20964, 2025. 1, 2, 3
[4] Richard J Chen, Ming Y Lu, Jingwen Wang, Drew FK
Williamson, Scott J Rodig, Neal I Lindeman, and Faisal
Mahmood. Pathomic fusion: an integrated framework for
fusing histopathology and genomic features for cancer diag-
nosis and prognosis.IEEE Transactions on Medical Imag-
ing, 41(4):757–770, 2020. 3
[5] Richard J. Chen, Ming Y . Lu, Wei-Hung Weng, Tiffany Y .
Chen, Drew F.K. Williamson, Trevor Manz, Maha Shady,
and Faisal Mahmood. Multimodal co-attention transformer
for survival prediction in gigapixel whole slide images. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), pages 4015–4025, 2021. 6
[6] Richard J Chen, Ming Y Lu, Wei-Hung Weng, Tiffany Y
Chen, Drew FK Williamson, Trevor Manz, Maha Shady, and
Faisal Mahmood. Multimodal co-attention transformer for
survival prediction in gigapixel whole slide images. InPro-
ceedings of the IEEE/CVF international conference on com-
puter vision, pages 4015–4025, 2021. 3, 6
[7] Justin J Choi, Jeanie Gribben, Myriam Lin, Erika L Abram-
son, and Juliet Aizer. Using an experiential learning model to
teach clinical reasoning theory and cognitive bias: an eval-
uation of a first-year medical student curriculum.Medical
Education Online, 28(1):2153782, 2023. 2
[8] Google DeepMind. Gemini 2.5 pro [large language model],
2025. Accessed 2025-11-14. 6, 7
[9] DeepSeek-AI et al. Deepseek-v3 technical report, 2025. 3
[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. Bert: Pre-training of deep bidirectional trans-
formers for language understanding. InProceedings of the
2019 conference of the North American chapter of the asso-
ciation for computational linguistics: human language tech-
nologies, volume 1 (long and short papers), pages 4171–
4186, 2019. 4
[11] Joseph H Donroe, Emilie Egger, Sarita Soares, Andre N So-
fair, and John Moriarty. Clinical reasoning: Perspectives
of expert clinicians on reasoning through complex clinical
cases.Cureus, 16(1), 2024. 2
[12] Dyke Ferber, Omar SM El Nahhas, Georg W ¨olflein, Is-
abella C Wiest, et al. Development and validation of an
autonomous artificial intelligence agent for clinical decision-
making in oncology.Nature cancer, pages 1–13, 2025. 3
[13] Fatemeh Ghezloo, Mehmet Saygin Seyfioglu, Rustin So-
raki, Wisdom O Ikezogwo, Beibin Li, Tejoram Vivekanan-
dan, Joann G Elmore, Ranjay Krishna, and Linda Shapiro.
Pathfinder: A multi-modal multi-agent system for medical

diagnostic decision-making applied to histopathology.arXiv
preprint arXiv:2502.08916, 2025. 1, 2, 3
[14] Bal ´azs Gy ˝orffy. Survival analysis across the entire transcrip-
tome identifies biomarkers with the highest prognostic power
in breast cancer.Computational and structural biotechnol-
ogy journal, 19:4101–4109, 2021. 1
[15] Frank E Harrell Jr, Kerry L Lee, and Daniel B Mark. Mul-
tivariable prognostic models: issues in developing models,
evaluating assumptions and adequacy, and measuring and re-
ducing errors.Statistics in medicine, 15(4):361–387, 1996.
6
[16] Maximilian Ilse, Jakub M. Tomczak, and Max Welling.
Attention-based deep multiple instance learning, 2018. 6
[17] Guillaume Jaume, Anurag Vaidya, Richard J Chen, Drew FK
Williamson, Paul Pu Liang, and Faisal Mahmood. Modeling
dense multimodal interactions between biological pathways
and histology for survival prediction. InProceedings of the
IEEE Conf. Comput. Vis. Pattern Recognit., pages 11579–
11590, 2024. 1, 3
[18] Xun Jiang, Feng Li, Han Zhao, Jiahao Qiu, Jiaying Wang,
et al. Long term memory: The foundation of ai self-
evolution.arXiv preprint arXiv:2410.15665, 2024. 2, 3
[19] Edward L Kaplan and Paul Meier. Nonparametric estima-
tion from incomplete observations.Journal of the American
statistical association, 53(282):457–481, 1958. 1, 6
[20] Yubin Kim, Chanwoo Park, Hyewon Jeong, Yik S Chan,
Xuhai Xu, Daniel McDuff, Hyeonhoon Lee, Marzyeh Ghas-
semi, Cynthia Breazeal, and Hae W Park. Mdagents: An
adaptive collaboration of llms for medical decision-making.
Advances in Neural Information Processing Systems, 37:
79410–79452, 2024. 6, 7
[21] G ¨unter Klambauer, Thomas Unterthiner, Andreas Mayr, and
Sepp Hochreiter. Self-normalizing neural networks.Ad-
vances in neural information processing systems, 30, 2017.
6, 7
[22] Hang Li, Fan Yang, Xiaohan Xing, Yu Zhao, Jun Zhang,
Yueping Liu, Mengxue Han, Junzhou Huang, Liansheng
Wang, and Jianhua Yao. Multi-modal multi-instance learning
using weakly correlated histopathological images and tabular
clinical information. InMedical Image Computing and Com-
puter Assisted Intervention – MICCAI 2021, pages 529–539,
Cham, 2021. Springer International Publishing. 6
[23] Junkai Li, Yunghwei Lai, Weitao Li, Jingyi Ren, et al. Agent
hospital: A simulacrum of hospital with evolvable medical
agents.arXiv preprint arXiv:2405.02957, 2024. 2, 3
[24] Rumeng Li, Xun Wang, Dan Berlowitz, Jesse Mez,
Honghuang Lin, and Hong Yu. Care-ad: a multi-agent large
language model framework for alzheimer’s disease predic-
tion using longitudinal clinical notes.npj Digital Medicine,
8(1):541, 2025.
[25] Jie Liu, Wenxuan Wang, Zizhan Ma, Guolin Huang, Yihang
SU, Kao-Jung Chang, Wenting Chen, Haoliang Li, Linlin
Shen, and Michael Lyu. Medchain: Bridging the gap be-
tween llm agents and clinical practice through interactive se-
quential benchmarking.arXiv preprint arXiv:2412.01605,
2024. 2, 3[26] Jinhui Lou, Yan Yang, Zhou Yu, Zhenqi Fu, Weidong
Han, Qingming Huang, and Jun Yu. Cxragent: Director-
orchestrated multi-stage reasoning for chest x-ray interpreta-
tion.arXiv preprint arXiv:2510.21324, 2025. 3
[27] Ming Y Lu, Drew FK Williamson, Tiffany Y Chen, Richard J
Chen, Matteo Barbieri, and Faisal Mahmood. Data-efficient
and weakly supervised computational pathology on whole-
slide images.Nature biomedical engineering, 5(6):555–570,
2021. 7, 3
[28] Xinheng Lyu, Yuci Liang, Wenting Chen, Meidan Ding, Ji-
aqi Yang, Guolin Huang, Daokun Zhang, Xiangjian He, and
Linlin Shen. Wsi-agents: A collaborative multi-agent system
for multi-modal whole slide image analysis. InInternational
Conference on Medical Image Computing and Computer-
Assisted Intervention, pages 680–690, 2025. 1, 2, 3, 5
[29] Nathan Mantel et al. Evaluation of survival data and two
new rank order statistics arising in its consideration.Cancer
Chemother Rep, 50(3):163–170, 1966. 6
[30] MyPathologyReport Team. My pathology report — pa-
tient pathology (traditional chinese).https://www.
mypathologyreport.ca/zh-TW/, 2025. Internation-
ally recognized, pathologist-written and peer-reviewed pa-
tient education resource; disclaimer: content is for general
information, not individualized medical advice. 5
[31] OpenAI. Gpt-5 (version 2025-08-07) [large language
model], 2025. Accessed 2025-11-14. 6, 7
[32] Ngoc Bui Lam Quang, Nam Le Nguyen Binh, Thanh-
Huy Nguyen, Le Thien Phuc Nguyen, Quan Nguyen, and
Ulas Bagci. Gmat: Grounded multi-agent clinical descrip-
tion generation for text encoder in vision-language mil for
whole slide image classification. InInternational Workshop
on Emerging LLM/LMM Applications in Medical Imaging,
pages 1–9. Springer, 2025. 1, 3
[33] Tim R ¨az, Aur ´elie Pahud De Mortanges, and Mauricio Reyes.
Explainable ai in medicine: challenges of integrating xai
into the future clinical routine.Frontiers in Radiology, 5:
1627169, 2025. 1, 3
[34] Paul Rutter. The importance of clinical reasoning in differen-
tial diagnosis for non-medical prescribers, nurses and phar-
macists.Clinics in Integrated Care, 31:100271, 2025. 2
[35] Andrew H Song, Richard J Chen, Tong Ding, Drew FK
Williamson, Guillaume Jaume, and Faisal Mahmood. Mor-
phological prototyping for unsupervised slide representation
learning in computational pathology. InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 11566–11578, 2024. 1, 3
[36] Yuxuan Sun, Yunlong Zhang, Yixuan Si, Chenglu Zhu,
Zhongyi Shui, Kai Zhang, Jingxiong Li, Xingheng Lyu, Tao
Lin, and Lin Yang. Pathgen-1.6m: 1.6 million pathology
image-text pairs generation through multi-agent collabora-
tion, 2024. 3
[37] Yuxuan Sun, Yixuan Si, Chenglu Zhu, Kai Zhang, Zhongyi
Shui, Bowen Ding, Tao Lin, and Lin Yang. Cpathagent:
An agent-based foundation model for interpretable high-
resolution pathology image analysis mimicking pathologists’
diagnostic logic.arXiv preprint arXiv:2505.20510, 2025. 1,
2, 3

[38] Xiangru Tang, Anni Zou, Zhuosheng Zhang, Ziming Li,
Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark Ger-
stein. Medagents: Large language models as collaborators
for zero-shot medical reasoning, 2024. 6, 7
[39] Yi Tang, Kaini Wang, Yang Chen, and Guangquan Zhou.
Endoagent: A memory-guided reflective agent for intelli-
gent endoscopic vision-to-decision reasoning.arXiv preprint
arXiv:2508.07292, 2025. 3
[40] Qwen Team et al. Qwen2.5 technical report, 2025. 3
[41] Eleftherios Tzanis, Lisa C Adams, Tugba Akinci
D’Antonoli, Keno K Bressem, Renato Cuocolo, Burak
Kocak, Christina Malamateniou, and Michail E Klontzas.
Agentic systems in radiology: Principles, opportunities,
privacy risks, regulation, and sustainability concerns.
Diagnostic and Interventional Imaging, 2025. 3
[42] Sheng Wang, Ruiming Wu, Charles Herndon, Yihang Liu,
Shunsuke Koga, Jeanne Shen, and Zhi Huang. Pathology-
cot: Learning visual chain-of-thought agent from expert
whole slide image diagnosis behavior.arXiv preprint
arXiv:2510.04587, 2025. 1, 2, 3
[43] Xiyue Wang, Junhan Zhao, Eliana Marostica, Wei Yuan, Ji-
etian Jin, et al. A pathology foundation model for cancer
diagnosis and prognosis prediction.Nature, 634(8035):970–
978, 2024. 4
[44] Jiwen Xin, Adam Mark, Cyrus Afrasiabi, Ginger Tsueng,
Moritz Juchler, et al. High-performance web services for
querying gene and variant annotation.Genome biology, 17
(1):91, 2016. 5
[45] Yingxue Xu and Hao Chen. Multimodal optimal transport-
based co-attention transformer with global structure con-
sistency for survival prediction. InProceedings of the
IEEE/CVF international conference on computer vision,
pages 21241–21251, 2023. 3, 6
[46] Jiawen Yao, Xinliang Zhu, Jitendra Jonnagaddala, Nicholas
Hawkins, and Junzhou Huang. Whole slide images based
cancer survival prediction using attention guided deep multi-
ple instance learning networks.Medical Image Analysis, 65:
101789, 2020. 6
[47] Wenchuan Zhang, Jingru Guo, Hengzhe Zhang, Penghao
Zhang, Jie Chen, Shuwan Zhang, Zhang Zhang, Yuhao Yi,
and Hong Bu. Patho-agenticrag: Towards multimodal agen-
tic retrieval-augmented generation for pathology vlms via re-
inforcement learning, 2025. 1, 2
[48] Yilan Zhang, Yingxue Xu, Jianqi Chen, Fengying Xie, and
Hao Chen. Prototypical information bottlenecking and dis-
entangling for multimodal cancer survival prediction.arXiv
preprint arXiv:2401.01646, 2024. 1, 3
[49] Fengtao Zhou and Hao Chen. Cross-modal translation and
alignment for survival analysis. InProceedings of the IEEE
Int. Conf. Comput. Vis., pages 21485–21494, 2023.
[50] Huajun Zhou, Fengtao Zhou, and Hao Chen. Cohort-
individual cooperative learning for multimodal cancer sur-
vival analysis.arXiv preprint arXiv:2404.02394, 2024. 6
[51] Junjie Zhou, Jiao Tang, Yingli Zuo, Peng Wan, Daoqiang
Zhang, and Wei Shao. Robust multimodal survival predic-
tion with the latent differentiation conditional variational au-
toencoder.arXiv preprint arXiv:2503.09496, 2025. 1, 3

SurvAgent: Hierarchical CoT-Enhanced Case Banking and
Dichotomy-Based Multi-Agent System for Multimodal Survival Prediction
Supplementary Material
Abstract.Appendix A presents the performance com-
parison betweenSurvAgentand existing open-source
pathology-specific multi-agent frameworks on the survival
prediction task. Appendix B provides the full WSI Attribute
Checklist, including detailed medical definitions for each
attribute and their relevance to patient prognosis. Appendix
C showcases two complete inference case studies, visualiz-
ingSurvAgent’s step-by-step analysis over WSI and ge-
nomic data and its final reasoning process. Appendix D
details the structure and contents of the CoT Case Bank.
Appendix E includes all core prompt designs used inSur-
vAgent. Appendix F describes the implementation details
and experimental settings used to construct and evaluate
SurvAgent. The source code will be released for public
access.
A. Comparison with Pathology Multi-Agent
Framework
We evaluatedSurvAgentagainst pathology-specific
multi-agent frameworks, WSI-Agent [28] and Patho-
AgenticRAG [47], across five TCGA cancer cohorts (Ta-
ble 1). Existing frameworks demonstrate limited perfor-
mance in survival prediction, with overall C-indexes of
0.524 and 0.509, and particularly low values on challeng-
ing cohorts such as GBMLGG (0.39 and 0.527). In con-
trast,SurvAgentconsistently achieves superior prognos-
tic accuracy across all datasets, with an overall C-index
of 0.713, improving 19.65% over existing methods and up
to 0.833 on GBMLGG, representing a gain of 30.6% over
Patho-AgenticRAG (0.527). These results highlight that our
multi-agent framework effectively integrates multi-modal
pathological information and substantially outperforms spe-
cialized pathology agents in survival prediction tasks.
B. WSI Attribute Checklist Overview
In this section, we provide a detailed illustration of the WSI
Attribute Checklist used inSurvAgent. Within our frame-
work, whole slide images (WSIs) serve as one of the pri-
mary modalities for downstream survival prediction, and
a central challenge lies in identifying which microscopic
pathological characteristics most strongly influence patient
prognosis. To address this, we construct a curated check-
list consisting of 16 prognostic histopathological attributes,
complemented by an additional global descriptor summa-
rizing the overall characteristics of the current slide. All
Tumor grade: High -grade
Depth of Invasion: Deep muscle invasion
Lymphovascular Invasion: Present
Perineural Invasion: Present
Lymph Node Metastasis: 1-2 nodes
Margin Status: Negative
Tumor Morphology: High -grade urothelial carcinoma……
Carcinoma in Situ: Absent
Variant Histology: Lymphoepithelioma -like
Squamous Differentiation: Focal
Glandular Differentiation: Extensive
Micropapillary Component: Unknown
Plasmacytoid Component: Unknown
Sarcomatoid Differentiation: Focal
Lymphocytic Infiltration: Mild
Necrosis Percentage: 10-30%
Summary: High -grade malignant neoplasm, perineural 
invasion, abundant lymphocytic infiltration, necrosis……WSI Atrribute Checklist
Figure 1. Overview of the WSI Attribute Checklist
attributes are automatically extracted from WSIs by PathA-
gent and compiled by the Search Agent, which aggregates
their definitions and clinical relevance based on pathology
guidelines and established medical literature. An example
of the full checklist is shown in Fig. 1, and the meaning and
definition of each attribute are described below.
Tumor Grade.A histological assessment of the tumor’s
cellular abnormality and proliferation rate. Reflects the in-
trinsic biological aggressiveness and growth potential of the
tumor. A powerful prognostic indicator; high-grade tumors
are consistently associated with significantly higher risks
of recurrence and progression to invasive disease, whereas
low-grade tumors typically follow a more indolent clinical
course.
Depth of Invasion.The extent to which the tumor infil-
trates the bladder wall, ranging from non-invasive mucosal
involvement to deep muscular penetration.Reflects the in-
trinsic aggressiveness and progression stage of the tumor.
One of the strongest predictors of survival; deeper invasion
is consistently associated with markedly worse overall sur-
vival.
Lymphovascular Invasion (LVI).Presence of tumor cells
within lymphatic or vascular channels. Indicates early

Table 1. The survival prediction performance (C-index) of the specialized pathological multi-agent framework in five TCGA cancer
research groups was compared. “*” indicates the best results from our reimplementation.
Model BLCA BRCA GBMLGG LUAD UCEC Overall
WSI-Agent* [28] 0.566±0.062 0.568±0.068 0.390±0.030 0.518±0.083 0.577±0.055 0.524
Patho-AgenticRAG* [47] 0.478±0.040 0.511±0.083 0.527±0.063 0.523±0.059 0.507±0.085 0.509
SurvAgent0.683±0.022 0.695±0.013 0.833±0.029 0.676±0.036 0.676±0.052 0.713
metastatic potential. LVI is a well-established independent
risk factor for distant metastasis and shortened survival.
Perineural Invasion (PNI).Tumor infiltration along or
around nerve fibers. Suggests highly invasive tumor be-
havior. Strongly associated with local recurrence and poor
long-term outcomes.
Lymph Node Metastasis.Involvement and number of
metastatic regional lymph nodes. Forms a crucial compo-
nent of TNM staging. Increasing nodal burden correlates
with stepwise reduction in survival probability.
Margin Status.Presence (positive) or absence (negative)
of residual tumor at the surgical resection margin. Positive
margins imply incomplete tumor removal. Strong predictor
of local relapse and reduced survival following surgery.
Tumor Morphology.Serves as the foundational visual
evidence for tumor grading and subtyping. The morpho-
logical assessment is critical; high-grade features, includ-
ing nuclear pleomorphism and frequent mitotic figures, are
directly correlated with aggressive clinical behavior, and
the identification of specific variant morphologies (e.g., mi-
cropapillary, sarcomatoid) carries significant prognostic and
therapeutic implications.
Carcinoma in Situ (CIS).A high-grade non-invasive flat
lesion with strong malignant potential. Often coexists with
aggressive invasive disease. The presence (especially ex-
tensive) of CIS predicts increased progression risk.
Variant Histology.Non-conventional morphological sub-
types such as squamous, glandular, micropapillary, sarco-
matoid, plasmacytoid, nested, or lymphoepithelioma-like
patterns. These variants frequently exhibit distinct biolog-
ical behaviors. Many variants, particularly micropapillary,
plasmacytoid, and sarcomatoid types, are associated with
highly aggressive disease and poor survival.
Squamous/Glandular Differentiation.Partial or exten-
sive differentiation toward squamous or glandular pheno-
types. Represents divergent tumor evolution. Extensive dif-
ferentiation is associated with advanced disease and inferior
clinical outcomes.
Micropapillary, Plasmacytoid, and Sarcomatoid Com-
ponents.Distinct morphologic components reflecting spe-
cific aggressive histologic variants. Indicate profound alter-
ations in tumor microarchitecture. These components are
widely recognized as markers of extremely poor prognosis.
Lymphocytic Infiltration.Degree of immune cell infiltra-tion within the tumor microenvironment (TME). Reflects
host immune response. Higher infiltration levels often cor-
relate with more favorable outcomes, whereas minimal in-
filtration suggests an “immune-cold” phenotype.
Necrosis Percentage.Proportion of necrotic tumor areas.
Indicates rapid tumor cell turnover and insufficient vascular
supply. Extensive necrosis is a known indicator of aggres-
sive tumor biology and poor survival.
Summary.A concise, free-text summarization that pro-
vides a comprehensive characterization of the current WSI
intended to mitigate limitations of fixed-value attribute sets.
Because the checklist attributes are discretized and selected
by prior feature-filtering steps, they may omit subtle, rare,
or composite histopathological cues and are susceptible
to selection or annotation bias. The Summary Attribute
is therefore designed to (1) capture additional prognostic
signals not well represented by the predefined categorical
fields, (2) record observations where multiple features in-
teract or where uncertainty exists, and (3) serve as a correc-
tive, interpretability-focused descriptor that complements
the structured attributes for downstream risk stratification.
C. Case Study Examples ofSurvAgent’s Rea-
soning Process
To further illustrate the detailed reasoning process ofSur-
vAgent, this section presents several representative case
studies. These examples demonstrate how the system an-
alyzes both WSI and genomic data, performs multi-level
prognostic reasoning, and integrates morphological cues
from WSIs with molecular signatures from gene profiles
to generate precise risk predictions. Fig. 4 illustrates the
core data involved in WSI analysis, gene analysis, CoT Case
Bank construction, and the inference pipeline for the patient
with case TCGA-XF-A9SU from the BLCA. Fig. 2 and
Fig. 3 focus on the inference stage, presenting SurvAgent’s
full WSI reports, gene reports, and the complete CoT-based
reasoning process.
D. Construction of the CoT Case Bank
To enable interpretable reasoning and retrieval-augmented
inference withinSurvAgent, we construct a unified Chain-
of-Thought (CoT) Case Bank. This repository stores struc-
tured reasoning trajectories across three complementary

levels: WSI-based analysis, gene-level analysis, and inte-
grated WSI–gene reasoning (Fig. 4). Each case follows a
standardized schema that includes the assigned risk level,
key evidence, and an explicit uncertainty statement, sum-
marizing both the essential prognostic cues and the inher-
ent ambiguity within the reasoning process. The Gene CoT
Case Bank captures reasoning grounded in genomic alter-
ations, abnormal expression patterns, and molecular signa-
tures, while the WSI–Gene CoT Case Bank consolidates
these perspectives into a coherent, cross-modal prognostic
analysis.
E. Core Prompt Configuration of SurvAgent
In this section, we present the core prompt design ofSur-
vAgent. Fig. 5–8 illustrate the prompt configurations used
in the key stages ofSurvAgent. Figure 5 presents the
prompt used by PathAgent to extract a structured WSI re-
port based on the predefined WSI Attribute Checklist. Fig-
ure 6 shows the prompt used by GeneAgent to perform sta-
tistical feature analysis and key gene selection across six
categories of functional genes, with tumor suppressor genes
illustrated as an example. Figure 7 displays the prompt used
by the Inference Agent to predict the exact survival time
within the coarse survival interval determined in the first-
stage reasoning, leveraging retrieved similar cases and the
current patient’s summarized reports. Figure 8 presents the
prompt used by the Inference Agent in the first-stage in-
ference, where it integrates retrieved analogous cases, the
patient’s WSI and gene reports, and predictions from multi-
ple expert survival models to determine the coarse survival
interval.
F. Implementation Details
F.1. Experimental Setup and Computing Environ-
ment
SurvAgent does not require any additional training, and all
results are obtained purely during inference. Experiments
are conducted on a computation node equipped with4×
NVIDIA RTX A6000 GPUs (48 GB each) and an Intel(R)
Xeon(R) Gold 6430 CPU. For all expert survival prediction
models used in our framework, WSI features are extracted
using CLAM [27] with the patch level set to 1, and all hy-
perparameters strictly follow the default settings provided
in its open-source implementation. The source code will be
released for public access.
F.2. SurvAgent Architecture
The SurvAgent framework is entirely developed in-house
without relying on any existing agent frameworks. Sur-
vAgent consists of four specialized agents: the Search
Agent, PathAgent, GenAgent, and Inference Agent. Their
implementations are described below.Search Agent is built upon DeepSeek-V3.2 [9], leverag-
ing its web-access capability in combination with a curated
pathology knowledge base to generate an initial WSI At-
tribute Checklist. The resulting checklist is subsequently
reviewed and refined by board-certified pathologists.
PathAgent employs PathGen-LLaV A [36] and Qwen2.5-
32B-Instruct [40] as its backbone models. PathGen-LLaV A
is responsible for producing expert-level pathological de-
scriptions from WSI image patches, while Qwen2.5-32B-
Instruct converts these descriptions into structured patho-
logical attributes, integrates them into a unified report, and
performs self-critique on the generated chain-of-thought
(CoT) to ensure quality and consistency.
GenAgent is built on Qwen2.5-32B-Instruct, which gen-
erates structured gene-level summaries, analyzes statistical
properties of functional gene categories, and performs CoT
quality verification for gene-related reasoning.
Inference Agent is built on Qwen2.5-32B-Instruct,
which is used to perform the final coarse survival interval
prediction as well as the precise survival time estimation.
F.3. Implementation of the Hierarchical WSI CoT-
Enhanced Case Bank
In processing whole-slide images (WSIs), we first apply
CLAM at patch level 1 with a patch size of256×256to tile
the entire slide, followed by filtering background patches.
Patch-level cancer cell detection is then performed using the
CHIEF pathology foundation model to identify high-risk re-
gions. Based on the attention scores produced by CHIEF,
we apply a DBSCAN clustering procedure to aggregate spa-
tially concentrated high-risk patches, using an epsilon of 4
and a minimum cluster size of 10, thereby determining can-
didate regions of interest.
WSI examination is conducted in a hierarchical man-
ner across three magnifications:2.5×(patch level 3),10×
(patch level 2), and20×(patch level 1). After observing
each magnification, PathAgent integrates multi-scale infor-
mation to form the final WSI-level report used for down-
stream survival prediction. At2.5×, PathAgent directly de-
scribes the region. At10×and20×, subregions are gener-
ated by re-tiling the parent region using512×512windows.
Due to the large number of high-resolution patches, we per-
form CoSMining-based filtering.
For this process, the image-based Self-Path Similarity
Matrix is computed via cosine similarity between CHIEF-
extracted patch features, while the text-based Self-Path
Similarity Matrix is constructed by embedding PathAgent-
generated descriptions using the text-embedding-3-large
model and computing pairwise cosine similarity. After re-
moving highly redundant patches from both modalities, we
take the intersection to obtain the final set of informative
subregions.
During the transition from10×to20×, PathAgent au-

tonomously determines whether further magnification is
necessary. This behavior is enabled by prompt-based self-
reflection, prompting PathAgent to identify uncertainty or
ambiguity in its own outputs and decide whether higher-
magnification inspection is required.
In the CoT generation stage, PathAgent aligns the multi-
scale WSI report with the patient’s ground-truth risk cat-
egory and survival time to perform reverse reasoning,
yielding fine-grained chain-of-thought trajectories that map
pathological findings to survival outcomes. To ensure valid-
ity, PathAgent further conducts a verification pass in which
the CoT is reevaluated without revealing the ground truth,
enabling the agent to detect inconsistencies and revise the
CoT accordingly.
F.4. Implementation of the Gene-Stratified CoT-
Enhanced Case Bank
For processing genomic data, we follow prior work to cate-
gorize genes into six major functional groups: Tumor Sup-
pressor Genes, Oncogenes, Protein Kinases, Cell Differen-
tiation Markers, Transcription Factors, and Cytokines and
Growth Factors. Owing to the large number of genes and
genomic fragments, we first conduct statistical quantifica-
tion to capture the global expression patterns of each gene
category.
Specifically, genomic information is divided into DNA-
level structural variation data (CNV), RNA-level expres-
sion data (RNA-seq), and other special genomic fragments.
For RNA-seq expression profiles, the mean and median ex-
pression values of each gene are computed to character-
ize the overall expression distribution of the correspond-
ing gene class. For CNV data, we quantify the mutation
rate—defined as the proportion of samples exhibiting mu-
tations such as point mutations, insertions/deletions, or am-
plifications/deletions—to assess the overall structural vari-
ability within each gene category.
After statistical quantification, GenAgent performs a
high-level analysis of each gene class to understand its
global expression characteristics. GenAgent then gener-
ates a preliminary class-level gene expression report and
identifies specific genes that require additional inspection.
For these selected genes, their raw expression values are
retrieved, and GenAgent accesses biological function in-
formation via the Gene Knowledge Base constructed us-
ing the python library mygene*. By integrating the statisti-
cal summaries with functional gene annotations, GenAgent
performs a systematic, coarse-to-fine analysis of expression
behaviors, ultimately producing a detailed report for each
gene category.
The reports from all categories are then consolidated into
a unified genomic feature report, which serves as input for
downstream inference. The process for generating chain-
of-thought (CoT) explanations follows the same proceduredescribed in the previous subsection.
F.5. Implementation of the Dichotomy-Based Multi-
Expert Agent Inference Module
The inference process of SurvAgent is conducted in two
stages: (1) estimating a coarse survival interval, and (2) pre-
dicting the exact survival time. In the first stage, the reason-
ing of the Inference Agent is enhanced by both retrieval-
augmented generation (RAG) and the outputs of multiple
expert survival prediction models. In the second stage, for
fine-grained survival time prediction, only RAG-based re-
trieval information is used to augment the inference process.
After obtaining the patient’s integrated WSI report and
genomic report, we perform cosine-similarity–based re-
trieval over the previously constructed WSI–Gene CoT
Case Bank to identify the top three most similar historical
cases. Each retrieved case provides its WSI report, gene
report, and corresponding chain-of-thought (CoT) annota-
tions.
Following the definition of risk scores from expert
survival models, we map each model’s predicted risk
value into one of four risk strata based on quartiles.
These strata align with the target survival intervals: High
(0–12 months), High-intermediate (12–24 months), Low-
intermediate (24–36 months), and Low (36+ months). The
Inference Agent integrates the current patient’s multimodal
information, retrieved case evidence, and risk assessments
from multiple expert models to determine the patient’s final
risk stratum.
Once the coarse survival interval is established, we fur-
ther retrieve the actual survival times of similar cases. These
values are provided as additional prompts to the Inference
Agent to facilitate precise survival time prediction at a
monthly resolution.

WSI Summarized Report
Case id: TCGA -XF-A9SJ
Gene Summarized Report
Case id: TCGA -XF-A9SJ
The patient's gene cohort demonstrates significant genomic instability with widespread amplifications and deletions across multiple gene groups, including tumor
suppressors, oncogenes, protein kinases, celldifferentiation markers, transcription factors, and cytokines and growth factors .Notably, tumor suppressor genes such asRB1
and PTEN aredownregulated, while CDKN 2Ashows alow-level gain and upregulation .Oncogenes like FGFR 3and ERBB 2areamplified but downregulated, and MYC is
amplified with unknown expression levels .Protein kinases such asMET and AURKA areupregulated, while EGFR and BRAF aredownregulated .Key cell differentiation
markers like CDH 1,GATA 3,TP63,KRT20,and FOXA 1aredownregulated, indicating anaggressive and undifferentiated tumor state .Transcription factors E2F3and MYC are
amplified, while ARand FOXA 1aredownregulated .Cytokines and growth factors such asVEGFA, PDGFRA, IL6,FGF2,EGF, TNF, and IL8show varying levels ofdysregulation,
contributing totumor growth, angiogenesis, and immune evasion .Overall, these findings suggest acomplex and aggressive bladder cancer with impaired tumor
suppression and dysregulated signaling pathways .
Final Result
Case id: TCGA -XF-A9SJ
**Reason** :The current patient's genomic report indicates significant genomic instability with widespread amplifications and deletions across multiple gene groups,
including tumor suppressors and oncogenes .Notably, tumor suppressor genes such asRB1and PTEN aredownregulated, while oncogenes like FGFR 3and ERBB 2are
amplified and upregulated .The presence ofsarcomatoid and glandular differentiation, along with abundant lymphocytic infiltration and necrosis, further supports an
aggressive tumor phenotype .The historical cases (case 1,case 2,and case 3)allexhibit similar aggressive features, including sarcomatoid differentiation, high -grade
urothelial carcinoma, and significant genetic instability, with survival times ranging from 2.2to4.8months .Given thehigh -risk profile and theprovided survival time range
of0-12months, thecurrent patient's prognosis aligns closely with thehigh -risk category .
**Risk** :-0.5786
**Survival Time** :3.54months"Tumor grade" :"High -grade",
"Depth ofInvasion" :"There was 4instance ofmuscularis propria invasion .",
"Lymphovascular Invasion" :"Absent",
"Perineural Invasion" :"Absent",
"Lymph Node Metastasis" :"not occurred",
"Margin Status" :"Negative",
"Tumor Morphology" :"High -grade urothelial carcinoma with possible sarcomatoid differentiation,
adenocarcinoma with glandular differentiation, myxoid stroma, spindle -shaped cells,
pleomorphic nuclei, necrosis, invasive fungal infection, heterotopic bone formation,
Cryptococcus species, parasitic cystitis",
"Carcinoma inSitu" :"Unknow",
"Variant Histology" :"Sarcomatoid \nGlandular differentiation
Squamous differentiation \nUrothelial carcinoma",
"Squamous Differentiation" :"Present",
"Glandular Differentiation" :"Present",
"Micropapillary Component" :"Absent",
"Plasmacytoid Component" :"Absent",
"Sarcomatoid Differentiation" :"Present",
"Lymphocytic Infiltration" :"Abundant",
"Necrosis Percentage" :"The necrosis rate intheobserved area is30-50%"
"Summary" :"The pathology slide demonstrates amoderately topoorly differentiated adenosquamous carcinoma with glandular structures that moderately deviate
from normal architecture, infiltrating the muscularis propria without breaching the serosal surface .The tumor cells exhibit pleomorphism, hyperchromatic nuclei, and
cribriform patterns, consistent with adenocarcinoma .Additionally, adensely cellular area with spindle -shaped and epithelioid cells, suggestive ofaspindle cellneoplasm
such asleiomyosarcoma, ispresent .The surrounding stroma shows dense collagen deposition, fibroblasts/myofibroblasts, and adesmoplastic reaction .Notable features
include areas ofnecrosis, particularly inthelower right quadrant, and adense neutrophilic infiltrate indicative ofanacute inflammatory response, with abscess formation
inthecentral area .Psammoma bodies areobserved, consistent with papillary renal cellcarcinoma .The upper leftquadrant contains adipocytes and acluster ofcells with
a\"pearl necklace \"appearance, characteristic ofCryptococcus species, supporting adiagnosis ofCryptococcal infection .Chronic inflammatory processes with
lymphocytic and plasma cellinfiltrates arealso noted .Novascular invasion isobserved .Definitive diagnosis requires additional immunohistochemical staining and clinical
correlation ."
GT: 3.22 monthsFigure 2. Case studies ofSurvAgent, including WSI Summarized report, Gene Summarized report, Final Result, and Survival Time gt
(Case ID: TCGA-XF-A9SJ).

WSI Summarized Report
Case id: TCGA -G2-A2EL
"Tumor Grade" :"High -grade",
"Depth ofInvasion" :"There was 1instance ofMuscularis propria invasion",
"Lymphovascular Invasion" :"Present",
"Perineural Invasion" :"Absent",
"Lymph Node Metastasis" :"not occurred",
"Margin Status" :"Negative",
"Tumor Morphology" :"High -grade urothelial carcinoma with pleomorphic and hyperchromatic cell
clusters, necrosis, hemorrhage, and vascular invasion ;papillary and solid growth patterns with dense
cellularity and desmoplastic stroma ;small cellcarcinoma –likeareas with small round blue cells,
spindle -cellmorphology, and high mitotic activity ;irregular glandular and cystic architectures
formed byatypical epithelial cells with disrupted tissue organization ;infiltration ofpleomorphic
malignant cells with thrombus, necrosis, and hemorrhage ;granulomatous and inflammatory
responses with lymphoid infiltrates ;presence ofcalcium oxalate crystals ;features suggestive of
desmoplastic urothelial carcinoma, angiosarcoma, orlymphoma, alongside benign cystic and stratified
squamous epithelial components .",
"Carcinoma inSitu" :"Absent",
"Variant Histology" :"Nested \nSarcomatoid \nGlandular differentiation \nSquamous differentiation",
"Squamous Differentiation" :"Extensive",
"Glandular Differentiation" :"Focal",
"Micropapillary Component" :"Absent",
"Plasmacytoid Component" :"Absent",
"Sarcomatoid Differentiation" :"Present",
"Lymphocytic Infiltration" :"Abundant",
"Necrosis Percentage" :"10-30%"
"Summary" :"High -grade urothelial carcinoma with necrosis, hemorrhage, and vascular invasion ;papillary architecture with hyperchromatic nuclei ;small cell
carcinoma with hyperchromatic nuclei ;spindle cell morphology ;small round blue cell tumor ;chronic inflammation ;lymphocytic infiltrate ;dense lymphoid infiltrate,
necrosis, calcium oxalate crystals, granulomatous inflammatory response, adenocarcinoma, glandular differentiation, lymphocytic infiltration, disrupted tissue architecture,
dense infiltration ofatypical cells, pleomorphic nuclei, necrosis, hemorrhage, and thrombus, dense lymphocytic infiltration, reactive orinflammatory process, stratified
squamous epithelium, lymphoid neoplasm ;necrosis, acute inflammation, fungal hyphae (Aspergillus), parasitic structure, granulomatous reaction, disrupted epithelial
layer, inflammatory process with lymphocytic infiltration, necrosis, and foreign body reaction, desmoplastic reaction, possible angiosarcoma, lymphoid aggregates,
presence ofnecrosis, cellular atypia, and dense aggregation ofsmall, round, darkly stained cells ;possible carcinoma orlymphoma ;high cellularity, hyperchromatic nuclei,
and mitotic activity ."
Gene Summarized Report
Case id: TCGA -G2-A2EL
The patient's gene cohort demonstrates significant genetic aberrations indicative ofadvanced bladder cancer .Notably, there iswidespread overexpression oftumor
suppressor genes with key genes like RB1,CDKN 2A,and PTEN showing downregulation, while TP53exhibits alow-level gain .Oncogenes show alow-level gain incopy
numbers with increased transcriptional activity, though specific oncogenes like FGFR 3,EGFR, and ERBB 2aredownregulated .Protein kinases exhibit dysregulated states
with upregulation ofBRAF and MAPK 1,but downregulation ofEGFR, MET, and AKT1.Cell differentiation markers are generally downregulated, suggesting apoorly
differentiated tumor state .Transcription factors show significant dysregulation with E2F3upregulated and FOXA 1,FOXA 2,FOXL 1,and FOXL 2downregulated .Cytokines
and growth factors demonstrate acomplex pattern with some genes amplified and others downregulated, indicating reduced signaling activity inthemajority ofgenes .
These genetic abnormalities collectively suggest ahighly aggressive and complex tumor phenotype .
Final Result
Case id: TCGA -G2-A2EL
**Reason** :The current patient's genomic report and WSI findings indicate ahighly aggressive and complex tumor phenotype with significant genetic aberrations,
including widespread overexpression oftumor suppressor genes and dysregulated oncogenes .The presence oflymphovascular invasion and abundant lymphocytic
infiltration, along with ahigh -grade urothelial carcinoma with squamous and glandular differentiation, further supports anaggressive tumor behavior .The historical cases
provide valuable context :Case 1,despite having alow risk level, shows amore controlled tumor environment with chronic inflammation and fibrosis, which isnotthecase
forthe current patient .Case 2,with ahigh -intermediate risk level, exhibits asimilar aggressive phenotype but lacks the lymphovascular invasion noted inthe current
patient .Case 3,with ahigh risk level, closely aligns with thecurrent patient's aggressive genetic and morphological features, including carcinoma insitu and squamous
metaplasia .Given theauthoritative pathologists' provided survival time range of24-36months and thehistorical cases' survival times, thecurrent patient's prognosis is
likely tobewithin thelower end ofthis range due totheaggressive nature ofthetumor .
**Risk** :-3.3541
**Survival Time** :27.34months
GT: 26.91 monthsFigure 3. Case studies ofSurvAgent, including WSI Summarized report, Gene Summarized report, Final Result, and Survival Time gt
(Case ID: TCGA-G2-A2EL).

WSI CoT Case Bank
Case id: TCGA -CF-A3MGGene CoT Case Bank 
Case id: TCGA -CF-A3MGWSI-Gene CoT Case Bank
Case id: TCGA -CF-A3MG
Risk Level: High -intermediate
Key Evidence: 
-The presence of high -grade urothelial carcinoma 
with pleomorphic cells, disorganized architecture, 
necrosis, and hemorrhage indicates a more 
aggressive tumor type.
-The tumor morphology includes atypical cells 
forming irregular nests and clusters, pleomorphic 
nuclei, increased nuclear -to-cytoplasmic ratios, 
and areas of necrosis and hemorrhage, which are 
indicative of a higher risk.
-The necrosis percentage is between 10 -30%, 
which is significant and suggests a more 
aggressive tumor behavior.
Uncertainty Statement: 
-The depth of invasion is non -invasive, which is a 
positive factor, but the unknown status of 
lymphovascular invasion, perineural invasion, 
lymph node metastasis, margin status, tumor 
number, and tumor size introduces significant 
uncertainty.
-The presence of glandular differentiation and 
moderate lymphocytic infiltration also adds 
complexity to the risk assessment.
Survival Time: 12.12 monthsRisk Level: High -intermediate
Key Evidence: 
The patient's genetic profile demonstrates 
significant aberrations that contribute to an 
aggressive bladder cancer phenotype. Notably, the 
deep deletion and downregulation of CDKN2A, 
along with the upregulation of RB1, suggest a loss of 
tumor suppressor function and increased cell 
proliferation. Additionally, the overexpression or 
amplification of oncogenes such as FGFR3, MYC, 
CCND1, ERBB2, and PIK3CA further drive 
uncontrolled cell proliferation. The downregulation 
of cell differentiation markers (GATA3, TP63, KRT5, 
and HOXA13) indicates an undifferentiated and 
aggressive tumor. These genetic alterations 
collectively suggest a highly aggressive tumor, but 
the presence of some downregulated pro -
tumorigenic signaling pathways (PDGFRA, IL6, and 
FGF2) may slightly mitigate the aggressiveness, 
leading to a high -intermediate risk level.
Uncertainty Statement: 
None.
Survival Time: 12.12 monthsRisk Level: High -intermediate
Key Evidence: 
1. **Tumor Morphology**: The report indicates a high -grade 
urothelial carcinoma with pleomorphic cells, disorganized 
architecture, necrosis, and hemorrhage, which are indicative of 
an aggressive tumor.
2. **Genetic Aberrations**: The gene report highlights 
significant genetic alterations, including deep deletion and 
downregulation of CDKN2A, upregulation of RB1, and 
downregulation of PTEN. Additionally, overexpression or 
amplification of oncogenes such as FGFR3, MYC, CCND1, ERBB2, 
and PIK3CA, along with upregulation of protein kinases like 
AKT1 and MET, suggest a highly aggressive and 
undifferentiated tumor phenotype.
3. **Necrosis Percentage**: The presence of necrosis in 10 -30% 
of the tumor indicates a rapidly growing and poorly 
vascularized tumor, which is often associated with a worse 
prognosis.
Uncertainty Statement: 
1. **Depth of Invasion**: The report states that the tumor is 
non-invasive, which is a positive factor. However, the status of 
lymphovascular invasion, perineural invasion, lymph node 
metastasis, margin status, tumor number, and tumor size are 
unknown, which could significantly impact the risk level.
2. **Carcinoma in Situ and Variant Histology**: The presence 
of carcinoma in situ and the specific variant histology are 
unknown, which could provide additional information on the 
aggressiveness of the tumor.
3. **Squamous Differentiation, Glandular Differentiation, 
Micropapillary Component, Plasmacytoid Component, and 
Sarcomatoid Differentiation**: These factors are also unknown, 
and their presence could further influence the risk level.
Survival Time: 12.12 monthsFigure 4.SurvAgent’s CoT Case Bank
-You are given multiple **region -level descriptions** from the same Whole Slide Image (WSI) of a bladder cancer patient.
-Given multiple **region -level descriptions** from the same Whole Slide Image (WSI), extract and aggregate **histopathological feature factors** into a single structured JSON output that represents the **overall WSI -level 
characteristics**, according to the provided **Feature Factor**.
-Please make sure to pay particular attention to the descriptions of certain characteristics that affect the survival time of the patients.
-These structured features will later be used to predict the patient’s **overall survival time**, classified into:
* A: 0 –1 year
* B: 1 –2 years
* C: 2 –3 years
* D: more than 3 years
-Please only output the final extracted JSON data, without including any other content.
Bladder Cancer Pathology Report:
{wsi_report}
List of features to extract: Tumor grade, Depth of Invasion, Lymphovascular Invasion, Perineural Invasion, Lymph Node Metasta sis, Margin Status, Tumor Morphology, Carcinoma in Situ, Variant Histology, Squamous 
Differentiation, Glandular Differentiation, Micropapillary Component, Plasmacytoid Component, Sarcomatoid Differentiation, Ly mph ocytic Infiltration, Necrosis Percentage, Summary
Please output strictly in the following JSON format, without any additional content:
{
"Tumor grade": "High -grade/Low -Grade",
"Depth of Invasion": "Non -invasive/Lamina propria invasion/Muscularis propria invasion/Deep muscle invasion/Unknown",
"Lymphovascular Invasion": "Present/Absent/Unknown",
"Perineural Invasion": "Present/Absent/Unknown",
"Lymph Node Metastasis": "None/1 -2 nodes/3 -10 nodes/>10 nodes/Unknown",
"Margin Status": "Negative/Positive/Unknown",
"Tumor Morphology": "...",
"Carcinoma in Situ": "Present/Absent/Extensive/Unknown",
"Variant Histology": "Urothelial carcinoma/Squamous differentiation/Glandular differentiation/Micropapillary/Plasmacytoid/Sar comatoid/Nested/Lymphoepithelioma -like/Unknown",
"Squamous Differentiation": "Present/Absent/Focal/Extensive/Unknown",
"Glandular Differentiation": "Present/Absent/Focal/Extensive/Unknown",
"Micropapillary Component": "Present/Absent/Focal/Extensive/Unknown",
"Plasmacytoid Component": "Present/Absent/Focal/Extensive/Unknown",
"Sarcomatoid Differentiation": "Present/Absent/Focal/Extensive/Unknown",
"Lymphocytic Infiltration": "Minimal/Mild/Moderate/Abundant/Unknown",
"Necrosis Percentage": "None/<10%/10 -30%/30 -50%/>50%/Unknown",
"Summary": " Important descriptions that may affect the duration of survival (Concise medical professional terms)"
}
Extraction Rules:
1. Extract strictly based on the report content, do not infer or guess
2. If a feature is not explicitly mentioned in the report, use "Unknown"
3. Ensure all feature values use predefined options
4. Output must be in valid JSON format
5. Pay special attention to bladder cancer -specific pathological features such as carcinoma in situ, variant histology, etc.
Please begin extraction:
Figure 5. The prompt for extracting a structured WSI report using the WSI attribute checklist.

## Task
-Below is a description of a patient's basic information, along with statistical sequencing data for the Tumor Suppressor Gene sgroup. Please describe the status of this gene group at the genomic level 
and indicate whether any abnormalities are present.
-The background knowledge describes the significance of this gene group. You can incorporate this background and your own know ledge of genetics into your analysis.
-In your analysis report, employ medical professional terminology as much as possible for analysis and responses.
-Your output should be as concise as possible! At the same time, it should be comprehensive in analysis, including all the abn ormal analyses.
## Background Knowledge
-**Significance of the Tumor Suppressor Genes Group**: In normal cells, tumor suppressor genes are active and function to inhi bit cell proliferation. When these genes are suppressed under certain 
conditions, or when their sequences are lost, their ability to inhibit proliferation is removed. This allows activated oncoge nesto function, leading to abnormal cell proliferation and potentially cancer. 
In cell cycle regulation, they prevent cells from progressing from one stage to the next, ensuring normal cell division.
-This gene group contains thousands of specific quantified gene sequences. To facilitate your analysis, I will provide their s tatistical information.
-Genes in this group can be broadly divided into two parts:
1.**Gene Copy Number Variation (CNV)**: CNV refers to a phenomenon where the copy number of a specific DNA sequence in the ge nom e changes relative to the normal reference genome. In normal 
human cells, most genes are biallelic (2 copies). CNV can lead to an increase (amplification) or decrease (deletion) in the c opy number of that segment.
*Values are typically integers:
*`-2` = Deep deletion
*`-1` = Heterozygous deletion (loss of heterozygosity)
*`0` = Normal copy number
*`1` = Low -level gain (gain)
*`2` = High -level amplification (amplification)
2.**RNA -seq Characterization**: RNA -seq uses high -throughput sequencing technology to measure the transcription level (i.e., mRN A abundance) of each gene within a cell. A high expression level 
indicates active transcription, while a low level indicates low transcriptional activity.
*Values are floating -point numbers:
*Positive numbers = Expression higher than the mean
*Negative numbers = Expression lower than the mean
## Tumor Suppressor Genes Group Statistics
{Mutation_situation}
--------------------------------------
{Expression_level_situation}
## Output Example
**Abnormal conditions and clinical effects**:
1.....
...
n.....
**The specific names of the genes that require detailed analysis**:
-TP53
-...
-others genes ... (etc.)Figure 6. The prompt for statistical feature analysis of six categories of functional genes and key gene selection, using tumor suppressor
genes as an example.
## Task
-Your current task is to predict the future survival time of a bladder cancer patient utilizing the genomic report. You are re quired to apply your expertise in pathology to accomplish this task.
-You will receive a detailed description of the patient’s genomic test results, along with three historical cases retrieved fr om the case database that most closely align with the current patient’s profile. 
Based on the current patient’s report and these historically relevant cases, conduct a thorough analysis to predict the patie nt’s future survival time.
-Meanwhile, for the current case, authoritative pathologists have provided the survival time range for this case. Please use t he given information to make the most accurate prediction (to the extent of 
specific numbers) for the survival time of the current case.
-Your output must consist of an analysis followed by the predicted survival time. Ensure your analysis is concise, clear, and accurate, presented as a coherent paragraph.
-In order to more accurately distinguish the survival time of patients, please retain the predicted survival time to two decim al places.
## Current Patient Report
**Genomic Report**
{gene_info }
**WSI Report**
{wsi_info}
## The most likely range of survival time
**{time_range} months**
## Historically Relevant Cases
"""
case1:
{rag1}
case2:
{rag2}
case3:
{rag3}
"""
## Output Example
**Reason**: ......
**Survival Time**: ......
Figure 7. The prompt for the Inference Agent to predict exact survival time within the identified interval in coarse survival intervals by
retrieved cases and summarized reports.

## Task
-Your current task is to predict the future survival time range of a bladder cancer patient using Whole Slide Images (WSI) and genomic reports. You need to apply expertise in pathology to complete 
this task.
-You will receive detailed descriptions of the patient's whole slide images and genetic test results, along with three histori cal cases retrieved from the case database that most closely match the 
current patient's condition. Additionally, you will have access to opinions from four pathology experts who only provide thei r final conclusions for your reference. Conduct a comprehensive analysis 
based on the current patient's report and these historically relevant cases to predict the patient's future survival time ran ge.
## Current Patient Report
{patient_info}
**WSI Report**
{wsi_info}
**Genomic Report**
{gene_info}
## Reference Information
### Expert Opinions
Expert 1: {CCL_class}
Expert 2: {MCAT_class}
Expert 3: {MOTCAT_class}
High: Prognostic survival time of 0-12 months
High -intermediate: Prognostic survival time of 12-24 months
Low -intermediate: Prognostic survival time of 24-36 months
Low: Prognostic survival time of over 36 months
### Historically Relevant Cases
"""
Case 1:
{rag1}
Case 2:
{rag2}
Case 3:
{rag3}
"""
## Please conduct a comprehensive diagnosis based on the following three aspects of information:
### Analysis Phase 1: Independent Feature Analysis
Based solely on the current case features, without considering other information, what is your preliminary judgment and why?
### Analysis Phase 2: Expert Opinion Evaluation
-What is the distribution of expert opinions?
-Are there significant disagreements?
### Analysis Phase 3: Similar Case Reference
-What insights can be gained from the analysis approaches of similar cases?
-What are the key differences between the current case and similar cases?
-How do these differences affect the diagnosis?
### Analysis Phase 4: Conflict Resolution
If there are conflicts among the three information sources:
-Which information source should be prioritized and why?
-How to reconcile contradictions between different information sources?
**Must avoid the following cognitive biases:**
-×Simply following the majority expert opinion
-×Blindly imitating similar case conclusions
-×Relying on external information while ignoring feature analysis
## Output Example
```json
{
"analysis_breakdown": {
"feature_based_judgment": "Independent judgment based on features",
"expert_consensus_analysis": "Expert opinion analysis",
"case_similarity_insights": "Insights from similar cases",
"conflict_resolution_strategy": "Conflict resolution strategy"
},
"final_prediction": "High/High -intermediate/Low -intermediate/Low",
"key_decision_factors": [
"Key decision factor 1",
"Key decision factor 2",
"Key decision factor 3"
],
"reasoning_chain": "Complete description of the reasoning chain"
}
```Figure 8. The prompt for the Inference Agent to classify each case into coarse survival intervals by retrieved cases, summarized reports,
and predictions from multiple expert survival models.