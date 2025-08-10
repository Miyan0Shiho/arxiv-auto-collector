# ADSeeker: A Knowledge-Infused Framework for Anomaly Detection and Reasoning

**Authors**: Kai Zhang, Zekai Zhang, Xihe Sun, Jingmeng Nie, Qinghui Chen, Han Hao, Jianyuan Guo, Jinglin Zhang

**Published**: 2025-08-05 05:05:06

**PDF URL**: [http://arxiv.org/pdf/2508.03088v1](http://arxiv.org/pdf/2508.03088v1)

## Abstract
Automatic vision inspection holds significant importance in industry
inspection. While multimodal large language models (MLLMs) exhibit strong
language understanding capabilities and hold promise for this task, their
performance remains significantly inferior to that of human experts. In this
context, we identify two key challenges: (i) insufficient integration of
anomaly detection (AD) knowledge during pre-training, and (ii) the lack of
technically precise and conte-aware language generation for anomaly reasoning.
To address these issues, we propose ADSeeker, an anomaly task assistant
designed to enhance inspection performance through knowledge-grounded
reasoning. ADSeeker leverages a curated visual document knowledge base,
SEEK-MVTec&VisA (SEEK-M&V), which we construct to address the limitations of
existing resources that rely solely on unstructured text. SEEK-M&V includes
semantic-rich descriptions and image-document pairs, enabling more
comprehensive anomaly understanding. To effectively retrieve and utilize this
knowledge, we introduce the Query Image-Knowledge Retrieval-Augmented
Generation (Q2K RAG) framework. To further enhance the performance in zero-shot
anomaly detection (ZSAD), ADSeeker leverages the Hierarchical Sparse Prompt
mechanism and type-level features to efficiently extract anomaly patterns.
Furthermore, to tackle the challenge of limited in industry anomaly detection
(IAD) data, we introduce the largest-scale AD dataset, Multi-type Anomaly
(MulA), encompassing 72 multi-scale defect types across 26 Categories.
Extensive experiments show that our plug-and-play framework, ADSeeker, achieves
state-of-the-art zero-shot performance on several benchmark datasets.

## Full Text


<!-- PDF content starts -->

ADS EEKER : A K NOWLEDGE -INFUSED FRAMEWORK FOR
ANOMALY DETECTION AND REASONING
Kai Zhang
School of Control Science and Engineering
Shandong UniversityZekai Zhang
School of Control Science and Engineering
Shandong University
Xihe Sun
School of Control Science and Engineering
Shandong UniversityJingmeng Nie
School of Computer Science
Shandong University
Qinghui Chen
School of Control Science and Engineering
Shandong UniversityHan Hao
School of Control Science and Engineering
Shandong University
Jianyuan Guo
Department of Computer Science
City University of Hong KongJinglin Zhang
School of Control Science and Engineering
Shandong University
August 6, 2025
ABSTRACT
Automatic vision inspection holds significant importance in industry inspection. While multimodal
large language models (MLLMs) exhibit strong language understanding capabilities and hold promise
for this task, their performance remains significantly inferior to that of human experts. In this context,
we identify two key challenges: (i) insufficient integration of anomaly detection (AD) knowledge
during pre-training, and (ii) the lack of technically precise and context-aware language generation
for anomaly reasoning. To address these issues, we propose ADSeeker, an anomaly task assistant
designed to enhance inspection performance through knowledge-grounded reasoning. ADSeeker
leverages a curated visual document knowledge base, SEEK-MVTec&VisA ( SEEK-M&V ), which we
construct to address the limitations of existing resources that rely solely on unstructured text. SEEK-
M&V includes semantic-rich descriptions and image-document pairs, enabling more comprehensive
anomaly understanding. To effectively retrieve and utilize this knowledge, we introduce the Query
Image-Knowledge Retrieval-Augmented Generation ( Q2K RAG ) framework. To further enhance the
performance in zero-shot anomaly detection (ZSAD), ADSeeker leverages the Hierarchical Sparse
Prompt mechanism and type-level features to efficiently extract anomaly patterns. Furthermore, to
tackle the challenge of limited in industry anomaly detection (IAD) data, we introduce the largest-
scale AD dataset, Multi-type Anomaly ( MulA ), encompassing 72 multi-scale defect types across
26 Categories. Extensive experiments show that our plug-and-play framework, ADSeeker, achieves
state-of-the-art zero-shot performance on several benchmark datasets.
1 Introduction
Multimodal Large Language Models (MLLMs), such as GPT-4o [ et al, 2024 ] and LLaV A [ Li et al., 2024 ], have shown
significant promise across various intellectual human tasks, including writing, programming, and Industry Anomaly
Detection (IAD). In IAD tasks, anomaly reasoning is distinct from anomaly detection. Anomaly reasoning necessitates
a finer level of granularity and logical reasoning in the model’s responses. This distinction is vital in identifyingarXiv:2508.03088v1  [cs.IR]  5 Aug 2025

APREPRINT - AUGUST 6, 2025
Medical Industry
Q: ls there a defect in the picture? What is the type of 
defect? Where is the location of the defect? 
A: The image you provided appears is a CT scan of the brain , 
showing a region with increased density (white area) compared to 
the surrounding tissue. This suggests a hyperdense area , which 
could indicate a number of different conditions. The hyperdensity 
is located in the right hemisphere of the brain, specifically in the 
region near the lateral ventricle and possibly extending into the  
temporal lobe . 
Q: What are the cause of the defect and its possible consequences?
A: This suggests a hyperdense area, which can be indicative of several things such 
as hemorrhage, calcification, tumor  and infection . This could lead to 
neurological deficits , increased intracranial pressure, functional impairment. Q: ls there a defect in the picture? What is the type of 
defect? Where is the location of the defect? 
A: Yes, there is a defect in the picture. The type of defect is 
“broken_large ”, which means there is a large breakage or 
chip along the opening edges of the bottle, disrupting the 
smooth circular contour. The location of the defect can be seen 
in the bottom right corner of the bottle  in the picture.
Q: What are the cause of the defect and its possible consequences?
A: The causes of the defect could include manufacturing errors , such as 
improper handling during production, or material defects that make the bottle 
more susceptible to breaking. The possible consequences of this defect include 
compromise structural integrity, sharp irregular shapes, and missing 
material . These irregularities create safety risks. Furthermore, the exposed 
surfaces exhibit texture/color contrasts with intact areas, which might lead to 
product rejection during quality checks .
Figure 1: Industrial and medical image anomaly reasoning results of ADSeeker.
the roles of a quality inspector versus a domain expert. Recent studies have explored the use of MLLMs for IAD
tasks, including detection, description, analysis, etc., by training models such as AnomalyGPT [ Gu et al., 2023 ] and
Anomaly-OV [ Xu et al., 2025 ], to incorporate domain-specific knowledge for improved anomaly reasoning. Although
some models excel in zero-shot anomaly detection (ZSAD), part of them struggle to provide accurate defect analysis
and detailed object descriptions during the testing phase when engaging in anomaly reasoning. It should be noted that
training methods for IAD MLLMs may seriously compromise the pre-trained knowledge and generalization of the
model and the lack of IAD training data can lead to overfitting in the content of the answers. To tackle these issues,
we propose ADSeeker , which addresses the limitations of domain knowledge through the Query Image—Knowledge
Retrieval-Augmented-Generation (Q2K RAG).
ADSeeker achieves multimodal hybrid retrieval, seeking the knowledge document most relevant to the query image.
Specifically, the query image and knowledge document are encoded into the Key Embedding and the Lock Embedding
respectively. Subsequent multimodal retrieval is performed in the joint feature space between these representations. To
compensate for the lack of knowledge base in IAD domain, we establish the first-ever knowledge base SEEK-M&V
composed of visually rich industry anomaly documents. In SEEK-M&V , each knowledge document has specific
reference pages containing the defect type and the defect analysis. Moreover, we utilize DeepSeek-R1 [et al, 2025] to
generate semantic-rich descriptions and application scenario introductions to expand the content of SEEK-M&V . As the
saying goes, “a picture is worth a thousand words.” SEEK-M&V enriches knowledge representation by preserving
graphic information for each anomaly type, offering a valuable reference for anomaly reasoning. Compared to existing
training-based approaches, it significantly improves both efficiency and effectiveness.
Existing approaches have introduced learnable prompts to adapt models for ZSAD. However, object-level prompts ( e.g.,
identifying objects such as “wood" or “leather") are incapable of providing detailed anomaly patterns, and publicly
available AD datasets struggle with semantically aligning these patterns. To address these gaps, we introduce Multi-type
Anomaly (MulA), the largest-scale dataset for ZSAD, which includes 72 defect types across 8 scenarios. The key
features of MulA, encoded through the Q2K RAG framework, are clustered using centroid-based techniques aligned
with predefined cluster centers. Additionally, ADSeeker utilizes a Hierarchical Sparse Prompt (HSP) mechanism to
extract type-level features ( e.g., distinguishing between “scratch" or “hole" anomalies) and reduce hallucinations caused
by suspected anomaly regions.
Extensive experiments validate the effectiveness and efficiency of ADSeeker. For example, anomaly reasoning
experiments on MMAD [ Jiang et al., 2024 ] benchmark demonstrate that ADSeeker outperforms other general models.
We also compare ADSeeker with best-performing training-based methods in IAD domain. Additionally, experiments
on 12 datasets from both industrial and medical sectors highlight ADSeeker’s superiority in ZSAD tasks. ADSeeker
effectively extracts anomaly patterns at the type level, offering a novel approach to anomaly reasoning. To validate its
effectiveness, we performed extensive tests to visualize the impact of type-level features across a range of products. We
also conducted ablation studies and evaluated the performance-resource trade-offs within our ADSeeker.
Our main contributions are summarized as follows:
•We propose ADSeeker, a novel framework that utilizes Q2K RAG and AD Expert module for IAD tasks,
enhancing specialization and fine granularity in reasoning.
•AHierarchical Sparse Prompt (HSP) mechanism is introduced to extract general anomaly patterns, resulting
in substantial improvements in ZSAD performance.
2

APREPRINT - AUGUST 6, 2025
•We propose SEEK-M&V , the first visual document knowledge base for anomaly reasoning. In addition, we
introduce MulA, the largest-scale AD dataset to date, which addresses the challenge of limited IAD data the
lack of type-level annotations.
2 Related Work
MLLM for industry anomaly detection (IAD). Recent advances have demonstrated the potential of MLLMs in
IAD. For example, AnomalyGPT is directly fine-tuned on IAD datasets to inject domain knowledge into the base
model. Moreover, to effectively extract detailed feature from abnormal samples, Myriad [ Li et al., 2023 ] developed an
Expert Perception module, which is designed to harness the prior knowledge of vision experts. However, these models
are constrained by the scope of their training knowledge and exhibit limited domain generalization. Existing IAD
approaches rarely become a trustworthy automatic vision inspection assistant [ Lin et al., 2023 ,wen Dong et al., 2024 ,
Yao et al., 2024 ,Chen et al., 2023 ,Liu et al., 2023a ,Liu et al., 2024 ]. In contrast, building on the excellence of our
ADSeeker framework, the anomaly reasoning performance of base models have been notably enhanced, as shown in
Table 5.
Zero-Shot Anomaly Detection (AD). Recently, WinCLIP [ Jeong et al., 2023 ] makes an early attempt at introducing
VLMs into ZSAD tasks, which leverage CLIP [ Radford et al., 2021 ] to compute the similarities between patch embed-
dings and textual embeddings. On this basis, AnomalyCLIP [ Zhou et al., 2024 ] enhances the ZSAD performance of
CLIP by proposing object-agnostic prompt learning. To tackle the issue of non-adaptation to common anomaly patterns,
AdaCLIP [ Cao et al., 2024 ] proposes hybrid dynamic learnable prompts which could flexibly adapt to the query images.
However, existing ZSAD methods [ Gu et al., 2024 ,Cao et al., 2023c ,Oquab et al., 2023 ,Kirillov et al., 2023 ] cannot
extract the generalizable anomaly patterns due to the object-level prompts. In contrast, our proposed HSP module could
extract detailed features from the query image and learn semantic-rich information from type-level features, enhancing
its sensitivity to anomalous features.
3 Datasets and Knowledge Base
3.1 MulA: Multi-type Anomaly Dataset
Existing AD datasets mainly focus on providing object-level ( e.g., “wood", “leather", “plastic") information
[Bergmann et al., 2019a ,Zou et al., 2022 ,Jezek et al., 2021 ,?], while the type-level feature ( e.g., “crack", “hole",
“scratch") plays an essential role in extracting defect-region features from the query image. To this end, we propose
Multi-type Anomaly dataset (MulA), which contains 11,226 images across 26 categories, surpassing existing AD
datasets in scale and diversity. We collect the data from different scenarios, including industry, ordinary, and workshops,
bringing multi-type generalizable anomaly patterns.
To build a dataset suitable for training, we manually annotated each sample with a high-quality mask. While some
scenarios lack a sufficiently diverse set of defect types, others suffer from a shortage of normal (defect-free) samples.
To address the former, we introduce Gaussian noise to enrich the defect distribution. For the latter, existing datasets
often replicate a small number of normal samples, which leads to overfitting and hinders generalization in ZSAD.
Therefore, we utilize image composition and geometric transformation to process different anomaly samples, obtaining
normal samples with significant diversity. As shown in Figure 2(A) , MulA incorporates data across various scales,
encompassing grayscale, RGB, and X-ray, exhibiting a wider generalization in industry. Besides the treatment of
samples, we categorize industrial scenarios according to industry types. Furthermore, we compared classical datasets
from both medical and industrial fields, as depicted in Table 1, thus validating the effectiveness of MulA. A detailed
introduction to MulA can be found in Appendix Section 1.
Dataset Number Categories Modal
Visa 10,821 12 RGB
MVTec 5,354 15 RGB
MPDD 6,572 4 RGB
BTAD 2,830 3 RGB
DAGM 6,124 6 RGB
DTD 2,830 3 RGB
MulA 11,226 26 X-ray Grey RGB
Table 1: Comparison of MulA with large-scale industrial anomaly detection dataset.
3

APREPRINT - AUGUST 6, 2025
Figure 2: (A) Statistics of the MulA Dataset and (B) examples of SEEK-M&V Knowledge Base.
3.2 SEEK-M&V: IAD Knowledge Base
Over the years, there is a lack of knowledge base for RAG frameworks in the field of anomaly detection
[Zhao et al., 2024 ,Wang et al., 2025 ,Wu et al., 2025 ,Lee et al., 2025 ]. Moreover, current knowledge base tends to
focus on textual information, while image features play an essential role in anomaly detection [ Yu et al., 2024 ]. Building
upon these insights, we build the first visual document knowledge base SEEK-M&V in IAD tasks, as illustrated in
Figure 2(B).
To enhance the specialization of domain knowledge in the SEEK-M&V , we present detailed information about the
anomaly with the official documentation for some of the AD datasets, such as anomaly type, anomaly analysis, and
anomaly description. Meanwhile, we find production scenarios and application scenarios for various products to enrich
the background information. Our SEEK-M&V contains both textual expertise and visual features. With generalizable
forms, multimodal features and semantic-rich information, our SEEK-M&V has great potential in generalizing to other
domains.
4 Method
In this section, we describe our proposed ADSeeker framework for anomaly detection and reasoning tasks, as shown
in Figure 3. Recent studies reveal that pretrained MLLMs function as generalists, possessing a broad knowledge
but underperforming in specialized domains, such as IAD. Therefore, our goal is to introduce a knowledge-injected
framework designed to equip the generalist with the AD domain knowledge. This approach circumvents the need for
pre-training while preserving the generalization capacity of the original model. We utilize the publicly available MLLM
(Qwen2.5-VL [ Bai et al., 2025 ] ) as the backbone of ADSeeker and the CLIP model (VIT-L/14@336px) as the encoder
4

APREPRINT - AUGUST 6, 2025
EncoderQ2K RAG
LLM“Is there any anomaly 
 in the query image ? ”
“According to the
reference 
knowledge base... ”
KDE
A image of a normal  [obj]
A image of a broken  [obj] ,
With [cls] defect type.
Cluster
EncoderCLIP
Visual Input[cls]
Text Input
AD Expert 
Tokenizer
Cos-sim
 Cos-sim
Concat
MLP
Query Image（A）Framework （B）HSP 
CLIP
Text EncoderLayer S
Layer S+1
Vision 
Encoder
Stage  S
Stage S+1
HSP
HSP
Learnable 
Prompts
ResidualN X
Update
GradsLinear
Trained Modules Element -wise Summation
 Learnable Prompts Matrix Multiplication Frozen Modules
pE
p
tE
QI
n
tE
QI
QI
QK
QK
Loc
Vis
ansL
cE
L
pT
nT
n
kI
qE
LE
nR
nG

nP
Query 
Embedding
L
 Sparse optimization
fEFinal
Figure 3: The architecture of ADSeeker. It consists of two main knowledge-infused pathways: (1) The query image
and knowledge documents are fed to the Q2K RAG to retrieve high-relevant domain knowledge. (2) The AD Expert
integrates defect-region information and type-level features into semantic-rich visual tokens which will be passed into
LLM.
of the AD Expert module. The CLIP encoder aligns the visual information and the textual information to perform
anomaly detection, and the LLM is responsible for textual instruction processing and complex reasoning.
4.1 Framework Overview
The framework of ADSeeker is illustrated in Figure 3. We store the type-level feature in learnable textual prompts
(Tp, Tn), such as “A image with [ cls] defect type”, which will be fed to CLIP Text Encoder. Similar to AdaCLIP, we
leverage CLIP encoder to align the semantic-visual features. In the process of feature fusion, the visual embedding Ep
and textual embedding Etextracted by the CLIP Encoder will be fed to AD Expert module and then integrated into
the anomaly prior feature Ec. Moreover, we introduce the HSP mechanism to preserve crucial visual features in CLIP.
Specifically, learnable embeddings are leveraged to sparsify invalid information of query images layer by layer, while
they will be combined with standard tokens at the beginning of each stage.
In the process of knowledge retrieval, the query image Iqand knowledge base In
kwill be processed by the encoder to
generate key feature KQand lock feature L:
L={L0, L1, L2, ..., L n−1}. (1)
The Q2K RAG module utilizes the key feature KQto retrieve the most relevant domain knowledge Lans. With
the injection of Lansand anomaly prior feature Ec, ADSeeker could achieve remarkable performance on anomaly
reasoning.
The key features of MulA datasets undergo centroid-based clustering aligned with predefined cluster centers. Subse-
quently, key feature KQis matched to the optimal cluster centroid to generate learnable textual prompts Tp, Tn. Thanks
to the high-quality data and type-level features of MulA dataset, our model could automatically generate accurate
prompts and understand common defects among abnormal samples. Compared to commonly used textual prompts in
ZSAD tasks, like “ A photo of a [ obj]”, type-level prompts are generally more detailed and versatile in scope, as they
encompass condensed content.
4.2 AD Expert:Hybrid Visual Fusion
When performing IAD tasks, the original MLLMs process basic Visual tokens without additional prior knowledge, such
as localization and discriminatory information. To utilize anomaly prior features in the inference stage, we introduce the
AD Expert module that transforms fine-grained information from query image and textual prompts into visual tokens.
We derive the localization and discriminatory information by calculating the cosine similarities between patch embed-
dings Ep, and positive textual embeddings Ep
tand negative textual embeddings En
t. We define the anomaly localization
5

APREPRINT - AUGUST 6, 2025
information Loc∈Rx1×das follows:
Loc=Unsample (cos(Ep, En
t)
cos(Ep, Ep
t) +cos(Ep, En
t)). (2)
As illustrated in Figure 3, we set a neural network to convert query image IQinto visual embeddings V is∈Rx2×d,
which will be integrated with the anomaly information to form anomaly prior embeddings Ec={Loc, V is }. Given
the semantic-rich visual embeddings, the defect characteristics will help the model generate fine-grained anomaly
descriptions and enhance the image-level anomaly detection performance.
4.3 Q2K RAG:KDE-Sampled Multimodal RAG
Q2K RAG
EncoderLock
Features
C
Score Rank   
 [0.94,0.92,0.88,0.83,0.79 ，0.74，0.65 ...]
Softmax
Ideal：
Yet：To efficiently sample ：6
Gaussian 
Clusters：Samples：Chosen Samples：Need Samples
MLP
KDE
CC
n
kI
QICCosine Similarity
...
QK
L KDE-Sample
ansLKey
Feature
Figure 4: Illustration of the core architecture of Q2K RAG
Instead of utilizing textual domain knowledge, our framework leverages multimodal documents in order to equip
ADSeeker with AD domain knowledge. The lightweight pre-trained Q2K RAG aligns the projection spaces M=
{KQ, L}of key feature and lock features. We adopt a similar methodology to VisRAG. To measure the relevance
between the query image and the multimodal documents, we compute the similarity Sof the embeddings:
S(KQ, L) ={Si|cos(KQ, Li), Li∈L}. (3)
Then the multimodal similarity Swill be sorted to seek for the required domain knowledge Lans. As shown in Figure
4, in an ideal situation, the most relevant knowledge is most inspiring for the problem. We just need to retrieve the
top-K↓expertise. Extensive experiments demonstrate that our seek accuracy rate has reached 83%. However, there is
still some possibility that required domain knowledge Lansis only at a relatively high rank, which greatly increases
the difficulty of multimodal retrieval. Furthermore, SEEK-M&V is a knowledge base organized by categories, where
different defect types of the same kind of object have a high degree of similarity. The simple top-k strategy has difficulty
in retrieving product-related scenario information.
To tackle these issues, we introduce the KDE-Sample to seek unevenly distributed domain knowledge. Interestingly, the
similarities of the same information across categories (defect, workshops, etc ) basically obey Gaussian distribution:
N=
In∼ G(µn, σ2
n)|In∈ S	
, (4)
whereGrepresents a Gaussian distribution, with µn,σ2
nindicating the nthmean and variance. Given the sorted similarity
scores S, we leverage the Bayesian Gaussian Mixture Modeling to infer the best class number K and perform GMM
clustering for similarity scores:
GMM (S) =
i1, i2...ik|i∼ G(µ, σ2)
. (5)
After that, we calculate the probability density weights W∗through the kernel density estimate (KDE) algorithm. In
order to quantify the quality score of each cluster, we deflate the original distribution:
Wn=1
KX
m∈inexp(W∗(m)−max
nW∗(n)). (6)
During the subsequent dynamic retrieval process, we sort the samples in descending order of score for each cluster
and dynamically select the samples from each cluster based on the ratio Wn. After the selected individual reaches the
6

APREPRINT - AUGUST 6, 2025
Industrial Defects Medical Anomalies
ModelMVTecAD VisA BTAD MPDD BrainMRI HeadCT Br35HAverage
CLIP 74.1 66.4 34.5 54.3 73.9 56.5 78.4 62.6
WinCLIP 91.8 78.8 68.2 63.6 92.6 90.0 80.5 80.8
AnomalyCLIP 91.5 82.1 88.3 77.0 90.3 93.4 94.6 88.2
AdaCLIP 89.2 85.8 88.6 76.0 94.8 91.4 97.7 89.1
Anomaly-OV 94.0 91.1 89.0 81.7 93.9 97.6 95.5 91.8
ADSeeker 94.3 91.5 94.0 85.9 97.5 96.6 97.9 94.0
Table 2: Image-Level AUROC performance of existing ZSAD approaches. ADSeeker surpasses most AD expert models
on this subtask. The top-performing result is presented in bold, while the second-best result is underlined.
sampling threshold, we terminate the selection process. The textual and visual retrieval pipelines demonstrate varying
levels of performance for different features. Without adaptive sampling, the combined retrieval can become redundant.
KDE-Sample strategy optimizes performance relative to data distribution, underscoring the value of information value
in hybrid retrieval.
4.4 Hierarchical Sparse Prompt
Drawing inspiration from the attention distribution of human industry inspection, that defect region of query images
deserves the majority of attention rather than distributed evenly, we design a learnable hierarchical sparse prompt (HSP)
module that integrates compression awareness and sparse optimization. The HSP module also addresses issues of
computational inefficiency and vulnerability to noise. The design of this module is primarily affected by AnomalyCLIP,
AdaCLIP, and AnomalyGPT.
As illustrated in Figure 3 (B), given the query image IQ, the defect features of query image are passed through linear
layers to obtain the query embeddings Eq∈RB×d. Then we introduce learnable prompt embeddings El∈RK×dand
adaptive parameters Pi∈RB×Kwhich are then fed into the sparse optimization module. The prompt embeddings
Elextract essential features from the embeddings Eqthrough Niterative updates. At the nthround of iteration, the
current residual Rncan be obtained by:
Rn=Eq−PnEn−1
l, (7)
the update gradient Gncan be represented by Gn=−µ·ET
qPnEl, where µrepresents a pre-defined descent rate. To
ensure the best match between the final embeddings En
land the fundamental feature, we calculate the iterated learnable
conversion factor P∗
nby combining Iterative Soft Thresholding Algorithm and sparsification theory:
P∗
n=Sign
Pn−Gn
σmax(ET
lEl)Pn−Gn
σmax(ET
lEl)−λ
Gn
·El (8)
At the end of each iteration, the learnable embeddings will be automatically updated by back propagation. Specifically,
the variation of parameters is based on the loss function Lbetween the learnable embeddings and the query embeddings,
which can be expressed as:
L= min
P1
2∥Eq−P∗
nEl∥2
2+λ∥P∗
n∥1, (9)
the sparsity coefficient λin the above formula represents the drop ratio of the fundamental features of query image.
This mechanism for sparse optimization is designed to accommodate features derived from linear layers, yet the actual
features extracted remain totally blind to us. To tackle this issue, we pre-train the network for feature extraction in
stages to enhance its sensitivity to defect regions. The linear layers are trained to enhance sensitivity to defect regions.
The final embeddings Efare integrated to a certain depth during the forward pass in the vision and text encoder. As we
explained in the section 4.1, the type-level textual prompts will be concatenated with defect-region features to enhance
ADSeeker’s fine-grained understanding of the anomaly pattern. Extensive experiments have demonstrated the potential
of this module in bridging the gap between human anomaly inspector and ADSeeker.
5 Experiments
5.1 Zero-shot Anomaly Detection
This study compares the proposed ADSeeker with existing ZSAD approaches. We conduct extensive experiments on
publicly available AD benchmarks from industrial and medical domains. Specifically, for the industry domain, we
7

APREPRINT - AUGUST 6, 2025
utilize MVTec AD, VisA, BTAD, and MPDD datasets. In the medical domain, we re-masked the brain medical detection
datasets BrainMRI, HeadCT, and Br35H, making them qualified anomaly detection datasets. The anomaly detection
and localization performance of ADSeeker is evaluated by the Area Under the Receiver Operating Characteristic
Curve (AUROC). The experiment follows the Zero-shot setting, our ADSeeker is fine-tuned on MVTec AD to evaluate
the ZSAD performance on other datasets. When evaluating the model on MVTec AD, the corresponding train-set is
replaced by VisA. To ensure the experiment fairness, we stopped utilizing Q2K RAG to ensure test-set is not leaked
during test.
The results are presented in Table 2, our ADSeeker have achieved significant enhancement on most of the AD
benchmarks in both medical and industrial domains. The enhancement of our model mainly originates from the
Hierarchical Sparse Prompt mechanism, which enables the model to extract the defect-region features and learn more
fine-grained semantics for anomaly detection in interaction with learnable textual prompts. ADSeeker has emerged as
the top performer, boasting the highest average ranking. This achievement highlights its impressive ability to generalize
effectively across various domains.
5.2 Anomaly Reasoning & ADSeeker Framework
Model DS-MVTec LOCO VisA Average
VLM R1 72.01 58.47 63.61 64.69
Anomaly R1 81.99 70.55 63.27 71.93
VLM R1* 77.65 71.03 68.11 72.26
LLaV A-OV 70.13 63.21 59.05 64.13
Anomaly OV 73.99 65.39 70.02 69.80
LLaV A-OV* 82.77 70.31 67.55 73.54
Table 3: Anomaly reasoning performance comparison with and the best-performing training methods. * represents base
model under the Seek-Setting
With the powerful capability of retrieving relevant expertise and the anomaly prior feature from the AD Expert module,
ADSeeker has achieved great improvement in performing anomaly reasoning tasks. As is shown in Figure 2, ADSeeker
could accurately locate defects and point out the defect type. Its anomaly description could achieve the accuracy of
human anomaly inspectors.
We evaluated anomaly reasoning capability via accuracy on multiple-choice subtasks from the MMAD benchmark,
comprising four anomaly-related and two object-related subtasks. Under a default 1-shot setting, compared models
received a randomly selected normal sample as template alongside the query image. For ADSeeker and Seek-Setting
models, the retrieved knowledge document replaced this normal sample as the template. As shown in Table 5, ADSeeker
(i.e., Qwen2.5-VL under Seek-Setting) was compared against open-source MLLMs. And each model is also tested
under Seek-Setting equipped with our Q2K RAG and AD Expert module to validate the efficacy of the plug-and-
play framework. ADSeeker delivers a remarkable accuracy of 69.90%, excelling across various subtasks. And our
framework shows enhance mainly on defect classification, localization and Anomaly Discrimination. The injected
anomaly prior feature enhances defect localization through visual information fusion, while the retrieved knowledge
document—containing specific defect types—facilitates defect classification and anomaly discrimination.
ADSeekerLoRA Accuracy
Baseline Q2K RAG Expert
✓ (76.6,67.1)
✓ ✓ (78.6,68.2)
✓ ✓ (77.9,67.7)
✓ ✓ ✓ (82.8,71.4)
✓ 5 (80.0,71.0)
✓ 10 (72.8,59.7)
✓ 20 (40.9,33.2)
Table 4: Ablation Results of proposed modules and training settings. The ✓in each columns signifies module integration.
The number in “LoRA” column denotes we utilize LoRA to fine-tune LLM for that epochs.
8

APREPRINT - AUGUST 6, 2025
Model Scale SettingAnomaly Defect Object
AverageDiscrimination Classification Localization Description Analysis Classification Analysis
LLaV A-OV 7BVanilla 51.17 46.13 41.85 62.19 69.73 90.31 80.93 63.19
Seek-Setting 55.35 (+4.18) 49.86 (+3.73) 54.34 (+12.49) 57.17 (-5.02) 73.20 (+3.47) 92.07 (+1.76) 84.31 (+3.38) 66.61(+3.42)
LLaV A-NeXT 7BVanilla 57.64 33.79 47.72 51.84 67.93 81.39 74.91 59.32
Seek-Setting 64.10 (+6.46) 53.29 (+19.50) 58.13 (+10.41) 54.09 (+2.25) 71.20 (+3.27) 87.70 (+6.31) 73.20 (-1.71) 65.96(+6.64)
InternVL2 8BVanilla 59.97 43.85 47.91 57.60 78.10 74.18 80.37 63.14
Seek-Setting 63.26 (+3.29) 58.71 (+14.86) 49.45 (+1.54) 56.45 (-1.15) 82.73 (+4.63) 84.79 (+10.61) 80.93 (+0.56) 68.05(+4.91)
Qwen2.5-VL3BVanilla 51.10 41.07 44.87 61.43 75.99 86.54 79.58 62.94
ADSeeker Seek-Setting 58.17 (+7.07) 48.71 (+7.64) 53.67 (+8.80) 69.79 (+8.36) 80.36 (+4.37) 86.78 (+0.24) 82.25 (+2.67) 68.53(+5.59)
Qwen2.5-VL7BVanilla 55.35 49.86 54.34 57.18 73.20 92.07 84.31 66.62
ADSeeker Seek-Setting 68.62 (+13.27) 53.82 (+3.96) 55.57 (+1.23) 60.51 (+3.33) 77.40 (+4.20) 89.91 (-2.16) 83.48 (-0.83) 69.90(+3.28)
Table 5: Performance comparison of both ADSeeker and other open-source MLLMs in MMAD with the standard
1-shot setting. Notably, each model is also tested under Seek-Setting.
Pretraining remains the most common solution for anomaly reasoning. Recent best-performing training-based ap-
proaches (e.g., Anomaly-OV and Anomaly-R1) have achieved remarkable results in IAD tasks. To evaluate the
efficacy of our framework, we compare their anomaly reasoning performance under both original training strategies
and the Seek-Setting. Due to the limitation of SEEK-M&V content, we utilize part of the MMAD dataset to conduct
evaluation..As illustrated in Table 3, extensive experiments demonstrate that ADSeeker could outperform existing SOTA
IAD approaches without additional training. Notably, Anomaly-R1 achieves remarkable performance by adapting the
GRPO strategy to fine-tuning with a limited training set, the utilization of that strategy is suggested for future works.
5.3 Efficiency and Efficacy Analysis
To demonstrate the efficacy of each proposed module in ADSeeker, we conduct ablation experiments by comparing
the accuracy of ADSeeker in 2 subtasks among MMAD. We primarily focus on 3 aspects: the Q2K RAG, the AD
Expert module and the impact of LoRA fine-tuning on ADSeeker. Our team constructed a 43K small-scale instruction
tuning dataset, we collected domain knowledge from AD datasets and our SEEK-M&V content. The dataset content
contains most of the knowledge base content and is constructed using template examples. As shown in Table 4, the
full-configuration ADSeeker achieves optimal performance across tasks. Each module contributes uniquely to anomaly
reasoning: Q2K RAG retrieves domain knowledge, while the AD Expert module provides anomaly prior features for
localization and discrimination. Note that model capability peaks then declines with extended training epochs. And
small-scale training sets exacerbate overfitting and catastrophic forgetting of pretrained knowledge. Moreover, our Q2K
RAG framework surpasses LoRA fine-tuning in both generalizability and precision.
As is shown in Table 6, we calculate the Memory Usage and Inference Time of each module to demonstrate the
efficiency of our proposed framework. When deployed on LLMs of comparable scale, ADSeeker incurs modest
overhead: ≤27% additional memory consumption and ≤2saverage inference latency. This demonstrates strong
potential in real-time industrial applications. Furthermore, ADSeeker is a training-efficient framework, which shows
no need for large-scale pre-training and effectively reduces the consumption of computational resources. In general,
proposed as a plug-and-play framework, ADSeeker could achieve the trade-offs between performance and computational
resources.
Efficiency Q2K RAG Expert Base Model ADSeeker
Mem (GiB) 4.25±0.53 2.44±0.09 22.63±1.35 28.36±1.68
Time (s) 1.13±0.16 1.52±0.02 4.47±2.16 6.14±2.20
Table 6: Memory usage and inference time calculation of each module (mean±SD)
5.4 Influence of Type-level Feature
We conduct comprehensive evaluations to validate the efficacy of the type-level representation. On the left of the
Figure 5, we calculate the similarity of the patch tokens and different kinds of textual prompts. Experiments reveal
stronger alignment between normal samples and [obj] -only prompts, while anomalous samples exhibit peak similarity
to prompts containing both [cls]&[obj] . Furthermore, for defect samples, [cls] prompts demonstrate significantly
higher feature value than [obj] prompts. These findings empirically validate the efficacy of type-level information
9

APREPRINT - AUGUST 6, 2025
representation. Our cross-dataset experiments of different-level prompts efficacy reveals that type-level [cls] features
significantly outperform [obj] features in image-level anomaly detection tasks. However, [obj] features maintain a
noteworthy complementary role, as evidenced by the ablation study in Figure 5.
Norm
Wood
Hole
Wood
Hole
Leather
Scratch
Wood
Scratch
Leather[cls H ] [obj][cls H ]
[obj][cls S ]
[obj]
base [obj] [cls] [obj]&[cls]Image -level 
Auroc[cls S ]
7080
75859095
Figure 5: Visualization the efficacy of the type-level feature through the Similarity Matrix and Ablation Study
6 Conclusion
In this paper, we propose an anomaly task assistant ADSeeker, leveraging the multimodal Q2K RAG to seek for domain
knowledge. To equip ADSeeker with AD expertise, we establish the first visual document knowledge base SEEK-M&V
for MLLMs in IAD tasks. Additionally, the HSP module is proposed to extract type-level features and generate sparse
prompts to enhance the ZSAD performance. Our work delves into the potential application of multimodal RAG in
anomaly detection, introducing a knowledge-infused framework without additional training.
References
[Bai et al., 2023] Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. (2023). Qwen-vl:
A versatile vision-language model for understanding, localization, text reading, and beyond.
[Bai et al., 2025] Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong,
H., Zhu, Y ., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y ., Ye, J., Zhang, X., Xie, T., Cheng, Z.,
Zhang, H., Yang, Z., Xu, H., and Lin, J. (2025). Qwen2.5-vl technical report. ArXiv , abs/2502.13923.
[Bergmann et al., 2019a] Bergmann, P., Fauser, M., Sattlegger, D., and Steger, C. (2019a). Mvtec ad — a comprehen-
sive real-world dataset for unsupervised anomaly detection. In 2019 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) , pages 9584–9592.
[Bergmann et al., 2019b] Bergmann, P., Fauser, M., Sattlegger, D., and Steger, C. (2019b). Uninformed students:
Student-teacher anomaly detection with discriminative latent embeddings. 2020 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR) , pages 4182–4191.
[Bornmann et al., 2019] Bornmann, L., Wray, K. B., and Haunschild, R. (2019). Citation concept analysis (CCA)—a
new form of citation analysis revealing the usefulness of concepts for other researchers illustrated by two exemplary
case studies including classic books by Thomas S. Kuhn and Karl R. Popper.
[Cao et al., 2023a] Cao, T. T., Zhu, J., and Pang, G. (2023a). Anomaly detection under distribution shift. 2023
IEEE/CVF International Conference on Computer Vision (ICCV) , pages 6488–6500.
[Cao et al., 2023b] Cao, Y ., Xu, X., Sun, C., Cheng, Y ., Du, Z., Gao, L., and Shen, W. (2023b). Personalizing
vision-language models with hybrid prompts for zero-shot anomaly detection. IEEE Transactions on Cybernetics ,
55:1917–1929.
[Cao et al., 2023c] Cao, Y ., Xu, X., Sun, C., Cheng, Y ., Du, Z., Gao, L., and Shen, W. (2023c). Segment any anomaly
without training via hybrid prompt regularization.
[Cao et al., 2024] Cao, Y ., Zhang, J., Frittoli, L., Cheng, Y ., Shen, W., and Boracchi, G. (2024). AdaCLIP: Adapting
CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection , page 55–72. Springer Nature Switzerland.
[Chen et al., 2023] Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Muyan, Z., Zhang, Q., Zhu, X., Lu, L., Li,
B., Luo, P., Lu, T., Qiao, Y ., and Dai, J. (2023). Intern vl: Scaling up vision foundation models and aligning for
generic visual-linguistic tasks. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) ,
pages 24185–24198.
10

APREPRINT - AUGUST 6, 2025
[et al, 2025] et al, D.-A. (2025). Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.
[et al, 2024] et al, O. (2024). Gpt-4o system card.
[Fear, 2005] Fear, S. (2005). Publication quality tables in L ATEX.http://www.ctan.org/pkg/booktabs .
[Geiger and Meek, 2005] Geiger, D. and Meek, C. (2005). Structured variational inference procedures and their
realizations (as incol). In Proceedings of Tenth International Workshop on Artificial Intelligence and Statistics,
The Barbados. The Society for Artificial Intelligence and Statistics.
[Gerndt, 1989] Gerndt, M. (1989). Automatic Parallelization for Distributed-Memory Multiprocessing Systems . PhD
thesis, University of Bonn, Bonn, Germany.
[Goossens et al., 1999] Goossens, M., Rahtz, S. P., Moore, R., and Sutor, R. S. (1999). The Latex Web Companion:
Integrating TEX, HTML, and XML . Addison-Wesley Longman Publishing Co., Inc., Boston, MA, USA, 1st
edition.
[Gu et al., 2024] Gu, Z., Zhu, B., Zhu, G., Chen, Y ., Li, H., Tang, M., and Wang, J. (2024). Filo: Zero-shot anomaly
detection by fine-grained description and high-quality localization.
[Gu et al., 2023] Gu, Z., Zhu, B., Zhu, G., Chen, Y ., Tang, M., and Wang, J. (2023). Anomalygpt: Detecting industrial
anomalies using large vision-language models.
[Gundy et al., 2007] Gundy, M. V ., Balzarotti, D., and Vigna, G. (2007). Catch me, if you can: Evading network
signatures with web-based polymorphic worms. In Proceedings of the first USENIX workshop on Offensive
Technologies , WOOT ’07, Berkley, CA. USENIX Association.
[Gundy et al., 2008] Gundy, M. V ., Balzarotti, D., and Vigna, G. (2008). Catch me, if you can: Evading network
signatures with web-based polymorphic worms. In Proceedings of the first USENIX workshop on Offensive
Technologies , WOOT ’08, pages 99–100, Berkley, CA. USENIX Association.
[Gundy et al., 2009] Gundy, M. V ., Balzarotti, D., and Vigna, G. (2009). Catch me, if you can: Evading network
signatures with web-based polymorphic worms. In Proceedings of the first USENIX workshop on Offensive
Technologies , WOOT ’09, pages 90–100, Berkley, CA. USENIX Association.
[Hagerup et al., 1993] Hagerup, T., Mehlhorn, K., and Munro, J. I. (1993). Maintaining discrete probability distribu-
tions optimally. In Proceedings of the 20th International Colloquium on Automata, Languages and Programming ,
volume 700 of Lecture Notes in Computer Science , pages 253–264, Berlin. Springer-Verlag.
[Ham et al., 2024] Ham, J., Jung, Y ., and Baek, J.-G. (2024). Glocalclip: Object-agnostic global-local prompt learning
for zero-shot anomaly detection.
[Jeong et al., 2023] Jeong, J., Zou, Y ., Kim, T., Zhang, D., Ravichandran, A., and Dabeer, O. (2023). Winclip:
Zero-/few-shot anomaly classification and segmentation.
[Jezek et al., 2021] Jezek, S., Jonak, M., Burget, R., Dvorak, P., and Skotak, M. (2021). Deep learning-based defect
detection of metal parts: evaluating current methods in complex conditions. In 2021 13th International Congress
on Ultra Modern Telecommunications and Control Systems and Workshops (ICUMT) , pages 66–71.
[Jiang et al., 2024] Jiang, X., Li, J., Deng, H., Liu, Y ., Gao, B.-B., Zhou, Y ., Li, J., Wang, C., and Zheng, F. (2024).
Mmad: A comprehensive benchmark for multimodal large language models in industrial anomaly detection.
[Kirillov et al., 2023] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S.,
Berg, A. C., Lo, W.-Y ., Dollár, P., and Girshick, R. B. (2023). Segment anything. 2023 IEEE/CVF International
Conference on Computer Vision (ICCV) , pages 3992–4003.
[Lee et al., 2025] Lee, N., Brouwer, E. D., Hajiramezanali, E., Park, C., and Scalia, G. (2025). Rag-enhanced
collaborative llm agents for drug discovery. ArXiv , abs/2502.17506.
[Li et al., 2024] Li, B., Zhang, Y ., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Li, Y ., Liu, Z., and Li, C. (2024).
Llava-onevision: Easy visual task transfer. ArXiv , abs/2408.03326.
[Li et al., 2025] Li, C., Zhou, S., Kong, J., Qi, L., and Xue, H. (2025). Kanoclip: Zero-shot anomaly detection through
knowledge-driven prompt learning and enhanced cross-modal integration.
[Li et al., 2023] Li, Y ., Wang, H., Yuan, S., Liu, M., Zhao, D., Guo, Y ., Xu, C., Shi, G., and Zuo, W. (2023). Myriad:
Large multimodal model by applying vision experts for industrial anomaly detection. ArXiv , abs/2310.19070.
[Lin et al., 2023] Lin, Z., Liu, C., Zhang, R., Gao, P., Qiu, L., Xiao, H., Qiu, H., Lin, C., Shao, W., Chen, K., Han,
J., Huang, S., Zhang, Y ., He, X., Li, H., and Qiao, Y . J. (2023). Sphinx: The joint mixing of weights, tasks, and
visual embeddings for multi-modal large language models. ArXiv , abs/2311.07575.
[Liu et al., 2023a] Liu, H., Li, C., Li, Y ., and Lee, Y . J. (2023a). Improved baselines with visual instruction tuning.
2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 26286–26296.
11

APREPRINT - AUGUST 6, 2025
[Liu et al., 2024] Liu, H., Li, C., Li, Y ., Li, B., Zhang, Y ., Shen, S., and Lee, Y . J. (2024). Llava-next: Improved
reasoning, ocr, and world knowledge.
[Liu et al., 2023b] Liu, H., Li, C., Wu, Q., and Lee, Y . J. (2023b). Visual instruction tuning.
[Oquab et al., 2023] Oquab, M., Darcet, T., Moutakanni, T., V o, H. Q., Szafraniec, M., Khalidov, V ., Fernandez, P.,
Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y . B., Li, S.-W.,
Misra, I., Rabbat, M. G., Sharma, V ., Synnaeve, G., Xu, H., Jégou, H., Mairal, J., Labatut, P., Joulin, A., and
Bojanowski, P. (2023). Dinov2: Learning robust visual features without supervision. ArXiv , abs/2304.07193.
[R Core Team, 2019] R Core Team (2019). R: A language and environment for statistical computing.
[Radford et al., 2021] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A.,
Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. (2021). Learning transferable visual models from natural
language supervision.
[Roth et al., 2021] Roth, K., Pemula, L., Zepeda, J., Scholkopf, B., Brox, T., and Gehler, P. (2021). Towards total
recall in industrial anomaly detection. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR) , pages 14298–14308.
[Wang et al., 2025] Wang, Q., Ding, R., Chen, Z., Wu, W., Wang, S., Xie, P., and Zhao, F. (2025). Vidorag: Visual
document retrieval-augmented generation via dynamic iterative reasoning agents. ArXiv , abs/2502.18017.
[wen Dong et al., 2024] wen Dong, X., Zhang, P., Zang, Y ., Cao, Y ., Wang, B., Ouyang, L., Wei, X., Zhang, S.,
Duan, H., Cao, M., Zhang, W., Li, Y ., Yan, H., Gao, Y ., Zhang, X., Li, W., Li, J., Chen, K., He, C., Zhang, X.,
Qiao, Y ., Lin, D., and Wang, J. (2024). Internlm-xcomposer2: Mastering free-form text-image composition and
comprehension in vision-language large model. ArXiv , abs/2401.16420.
[Wu et al., 2025] Wu, Y ., Long, Q., Li, J., Yu, J., and Wang, W. (2025). Visual-rag: Benchmarking text-to-image
retrieval augmented generation for visual knowledge intensive queries. ArXiv , abs/2502.16636.
[Xie et al., 2023] Xie, G., Wang, J., Liu, J., Zheng, F., and Jin, Y . (2023). Pushing the limits of fewshot anomaly
detection in industry vision: Graphcore. ArXiv , abs/2301.12082.
[Xu et al., 2025] Xu, J., Lo, S.-Y ., Safaei, B., Patel, V . M., and Dwivedi, I. (2025). Towards zero-shot anomaly detection
and reasoning with multimodal large language models.
[Yao et al., 2024] Yao, Y ., Yu, T., Zhang, A., Wang, C., Cui, J., Zhu, H., Cai, T., Li, H., Zhao, W., He, Z., Chen, Q.-A.,
Zhou, H., Zou, Z., Zhang, H., Hu, S., Zheng, Z., Zhou, J., Cai, J., Han, X., Zeng, G., Li, D., Liu, Z., and Sun, M.
(2024). Minicpm-v: A gpt-4v level mllm on your phone. ArXiv , abs/2408.01800.
[You et al., 2022] You, Z., Cui, L., Shen, Y ., Yang, K., Lu, X., Zheng, Y ., and Le, X. (2022). A unified model for
multi-class anomaly detection. ArXiv , abs/2206.03687.
[Yu et al., 2024] Yu, S., Tang, C., Xu, B., Cui, J., Ran, J., Yan, Y ., Liu, Z., Wang, S., Han, X., Liu, Z., and Sun, M. (2024).
Visrag: Vision-based retrieval-augmented generation on multi-modality documents. ArXiv , abs/2410.10594.
[Yuan et al., 2025] Yuan, J., Jie, P., Zhang, J., Li, Z., and Gao, C. (2025). Mfp-clip: Exploring the efficacy of multi-form
prompts for zero-shot industrial anomaly detection.
[Zhao et al., 2024] Zhao, P., Zhang, H., Yu, Q., Wang, Z., Geng, Y ., Fu, F., Yang, L., Zhang, W., and Cui, B. (2024).
Retrieval-augmented generation for ai-generated content: A survey. ArXiv , abs/2402.19473.
[Zhou et al., 2008] Zhou, G., Lu, J., Wan, C.-Y ., Yarvis, M. D., and Stankovic, J. A. (2008). Body Sensor Networks .
MIT Press, Cambridge, MA.
[Zhou et al., 2010] Zhou, G., Wu, Y ., Yan, T., He, T., Huang, C., Stankovic, J. A., and Abdelzaher, T. F. (2010). A
multifrequency mac specially designed for wireless sensor network applications. ACM Trans. Embed. Comput.
Syst., 9(4):39:1–39:41.
[Zhou et al., 2021] Zhou, K., Yang, J., Loy, C. C., and Liu, Z. (2021). Learning to prompt for vision-language models.
International Journal of Computer Vision , 130:2337 – 2348.
[Zhou et al., 2024] Zhou, Q., Pang, G., Tian, Y ., He, S., and Chen, J. (2024). Anomalyclip: Object-agnostic prompt
learning for zero-shot anomaly detection.
[Zhu et al., 2024] Zhu, J., Ong, Y . S., Shen, C., and Pang, G. (2024). Fine-grained abnormality prompt learning for
zero-shot anomaly detection. ArXiv , abs/2410.10289.
[Zou et al., 2022] Zou, Y ., Jeong, J., Pemula, L., Zhang, D., and Dabeer, O. (2022). Spot-the-difference self-supervised
pre-training for anomaly detection and segmentation. arXiv preprint arXiv:2207.14315 .
12