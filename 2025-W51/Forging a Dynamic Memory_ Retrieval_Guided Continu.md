# Forging a Dynamic Memory: Retrieval-Guided Continual Learning for Generalist Medical Foundation Models

**Authors**: Zizhi Chen, Yizhen Gao, Minghao Han, Yizhou Liu, Zhaoyu Chen, Dingkang Yang, Lihua Zhang

**Published**: 2025-12-15 08:09:40

**PDF URL**: [https://arxiv.org/pdf/2512.13072v1](https://arxiv.org/pdf/2512.13072v1)

## Abstract
Multimodal biomedical Vision-Language Models (VLMs) exhibit immense potential in the field of Continual Learning (CL). However, they confront a core dilemma: how to preserve fine-grained intra-modality features while bridging the significant domain gap across different modalities. To address this challenge, we propose a comprehensive framework. Leveraging our 18-million multimodal and comprehensive medical retrieval database derived from PubMed scientific papers, we pioneer the integration of Retrieval-Augmented Generation (RAG) into CL. Specifically, we employ a multi-modal, multi-layer RAG system that provides real-time guidance for model fine-tuning through dynamic, on-demand knowledge retrieval. Building upon this, we introduce a dynamic knowledge distillation framework. This framework precisely resolves the aforementioned core dilemma by dynamically modulating the importance of the parameter space, the granularity of the distilled knowledge, and the data distribution of the reference dataset in accordance with the required level of detail. To thoroughly validate the clinical value of our strategy, we have designed a more rigorous \textbf{M}edical Generalist Task Incremental Learning (MGTIL) benchmark. This benchmark is engineered to simultaneously evaluate the model's capacity for adaptation to significant domain shifts, retention of subtle intra-domain features, and real-time learning of novel and complex medical tasks. Extensive experimental results demonstrate that our proposed method achieves state-of-the-art (SOTA) performance across all metrics. The code is provided in the supplementary materials.

## Full Text


<!-- PDF content starts -->

FUDAN
UNIVERSITYForging a Dynamic Memory: Retrieval-Guided Continual Learning for
Generalist Medical Foundation Models
Zizhi Chen1,2,∗, Yizhen Gao3,∗, Minghao Han1,2, Yizhou Liu1,2, Zhaoyu Chen1,
Dingkang Yang1,2,†,§, Lihua Zhang1,2,§
1College of Intelligent Robotics and Advanced Manufacturing, Fudan University
2Fysics Intelligence Technologies Co., Ltd. (Fysics AI)
3School of Computer Science and Engineering , Central South University
∗Equal contribution,†Team lead,§Corresponding author
Abstract
Multimodal biomedical Vision-Language Models (VLMs) exhibit immense potential in the field of Continual
Learning (CL). However, they confront a core dilemma: how to preserve fine-grained intra-modality features
while bridging the significant domain gap across different modalities. To address this challenge, we
propose a comprehensive framework. Leveraging our 18-million multimodal and comprehensive medical
retrieval database derived from PubMed scientific papers, we pioneer the integration of Retrieval-Augmented
Generation (RAG) into CL. Specifically, we employ a multi-modal, multi-layer RAG system that provides
real-time guidance for model fine-tuning through dynamic, on-demand knowledge retrieval. Building
upon this, we introduce a dynamic knowledge distillation framework. This framework precisely resolves
the aforementioned core dilemma by dynamically modulating the importance of the parameter space, the
granularity of the distilled knowledge, and the data distribution of the reference dataset in accordance with
the required level of detail. To thoroughly validate the clinical value of our strategy, we have designed a
more rigorousMedicalGeneralistTaskIncrementalLearning(MGTIL)benchmark. This benchmark
is engineered to simultaneously evaluate the model’s capacity for adaptation to significant domain shifts,
retention of subtle intra-domain features, and real-time learning of novel and complex medical tasks.
Extensive experimental results demonstrate that our proposed method achieves state-of-the-art (SOTA)
performance across all metrics. The code is provided in the supplementary materials.
Date:December 16, 2025
Corresponding:dicken@fysics.ai,chenzz24@m.fudan.edu.cn
Project Page:https://github.com/CZZZZZZZZZZZZZZZZZ/PRIMED
1 Introduction
Vision-Language Models (VLMs) such as CLIP [ 59] have succeeded in zero-shot and fine-tuning. In medicine,
PMC-CLIP [ 50], trained on million level image-text pairs from scientific articles, created the first Generalist Medical
Foundation Model, enabling multimodal AI to aid clinical diagnosis. Yet updating pre-trained VLMs and applying
them to many downstream tasks across diverse medical domains and difficulty levels is hard. Retraining from scratch or
keeping a separate fine-tuned model per task is often impractical due to onerous data curation and consolidation, scarce
data for many diseases, and ethical limits. These approaches also incur prohibitive compute and fit poorly with today’s
1arXiv:2512.13072v1  [cs.CV]  15 Dec 2025

medical data ecosystem. Continual Learning (CL) [ 48,61,83,90], which learns incrementally from task sequences,
offers a viable way to handle these constraints.
CatDogCowSheepHorseSpider
SquirrelChickenElephantButterfly
CT Endoscope Fundus
Cell Pathology Skin      Focal Subtlety           Heterogeneous Principles                      Sub-domain Discreteness    Global Polymorphism         Low-level Isomorphism                         Manifold Connectivity



Intra-Domain         Cross-Domain             Embedding Space Topology
   Medical Image            Natural Image
Patho:BLCA Patho:BRCA
Patho:LUAD Patho:LUSC
Patho
FundusCT
Skin
Cat:Blue Cat:SphynxCat:Maine Cat:Kinkalow Cat Dog
Horse Cow
Figure 1 Naturalvs.Medical: Natural images are intra-domain
diverse yet cross-domain similar. Medical images are intra-domain
uniform but cross-domain distinct.However, CL in medical scenarios presents a more
complex situation. As shown in Fig. 1, natural images
and medical images are fundamentally different. In the
natural image, intra-domain samples exhibit significant
morphological variability, while inter-domain samples
still share correlations at the level of low-level features.
Consequently, the overall data distribution tends to form
a mesh-like structure. In contrast, medical images dis-
play minimal intra-domain variation but are governed
by drastically different imaging principles across do-
mains, resulting in a clustered distribution. This implies
that in medical CL, we must simultaneously achieve
finer-grained intra-domain transfer and bridge a more
challenging inter-domain gap.
In addition, CL can lead tocatastrophic forget-
ting[ 74,78,79]. For continuously fine-tuned VLMs,
this manifests not only as forgetting previously learned downstream tasks but also as degradation of pretrained knowledge,
which significantly impairs their zero-shot capabilities. A straightforward approach to mitigate forgetting is to replay a
small subset of stored historical samples [ 61]. However, since pretrained data are typically inaccessible, this strategy
is impractical for VLMs. Consequently, CL for VLMs is often simulated by sampling [ 56,85,90] or synthesizing
data [ 27,54,76] based on ImageNet [ 20] labels to mimic the replay process. Nevertheless, constructing reference
datasets in a label-based manner is not straightforward for medical VLMs and their large-scale ecosystems, built around
contextual or caption-based supervision. Meanwhile, Retrieval-Augmented Generation (RAG) [ 46] has become an
essential component in modern LLM systems, improving reasoning performance by incorporating retrieved information
through keyword matching [ 63] or embedding-based similarity search [ 15,49,89]. This insight inspires a new approach:
we can construct reference datasets by generating pseudo-labeled content in bulk, using questions as carriers and
employing a RAG-based framework to enable contextual reference during learning. More importantly, compared with
previous approaches that rely on sampling or generation, RAG enables more fine-grained and dynamic querying. It
not only better captures the clustered distribution patterns of large-scale medical data but also dynamically adapts to
domain-internal shifts, cross-domain transitions, and complex scenarios involving difficult tasks.
Based on this, we propose thePrecisionRetrieval-Infused model forMEDical(PRIMED)framework. First, we
construct an18-millionmultimodal medical image retrieval database from PubMed scientific papers using a data-cleaning
pipeline and the Qwen3-embedding-8B model [ 89]. To achieve intra-modal fine-grained feature memory distillation, we
collect and organize approximately3,000general medical label entries as question pool, hierarchically refined from
perspectives such as domain, lesion location, and disease type, enabling precise disentanglement of clustered medical
knowledge across domains. During training, we introduce a dynamic distribution regulator that shifts the reference
dataset from uniform to real-time multimodal, updating the RAG query vector database’s content and domain mix
based on past and ongoing fine-tuning to create a personalized “review plan” that enables fine-grained intra-domain
specialization and precise inter-domain recall. To preserve the global structure of the embedding space and maintain the
model’s zero-shot capability, we introduceContrastiveKnowledgeTransfer(CKT)andCross-ModalityConsistency
(CMC)loss strategies. Finally, by applying a dynamic model weight importance feedback, we achieve compatibility
between global structural preservation and fine-grained personalized feature retention. To better evaluate our approach,
we proposeMedicalGeneralistTaskIncrementallearning(MGTIL)benchmark. Previous medical CL studies have
focused always on single-domain [ 28,35,70] or support for base model training [ 52,80], lacking comprehensive
cross-domain benchmarks. We design two evaluations:HieraMedTransfer, simultaneously simulating intra and inter
domain incremental learning across diverse tasks; andMedXtreme, evaluating the model’s continual memory after
fine-tuning on ultra-challenging medical classification tasks. Our contributions are summarized as follows:
•We present an 18M multimodal retrieval database and a 3,000 fine-grained medical question pool that enable real-time
fine-grained retrieval.
2

  Question Database        
 Question Pool 
a. Real-time Fine-Grained Retrieval b. Distillation and Alignment c. Dynamic Fisher Weight Guard
       Serrated  Adenoma
Task t
      Mild        Severe
Task NAdd
A pathology photo 
of serrated lesion in 
the colon Image     Caption   Dense Retrieval
LossGet
Embedding Model
12 3 Visual Rerank 
0.49 0.07
BM25 Gate
Colon Patho.
MapSerrated 
lesion
VLM
VLM
...4
Traditional serrated 
adenoma of colonic 
mucosa (sigmoid) 
(H&E 2X).VLM 
Image
EncoderText
EncoderPast Memory
Memory-based domain partitioning
VLM 
Image
EncoderText
Encoder Data
Domain
Task
        Covid       Normal
Task 1
         NGT-N     ETT-BTask 2
VLM 
 VLM 
Dynamic Retrieval DataDistillation Alignment
Domain Bag General BagDynamic Siphon
Fundus:ROP
gcardiac
Task Bag
serrated
Covid tissue Cell
Fisher Importance
Figure 2Framework overview of PRIMED. (a)Real-time Fine-Grained Retrieval. Perform multi-level retrieval between the
real-time query repository and the pre-constructed 18-million multimodal retrieval database, and obtain specialized image-caption
pairs through a Dynamic Siphon mechanism. (b)Distillation and Alignment. Contrastive Knowledge Transfer L𝐶𝐾𝑇 is used for
matching CLIP 𝜃𝑡and CLIP𝜃𝑡−1on our Retrieval data, while using CMC loss L𝐶𝑀𝐶 to preserve the modality alignment of CLIP 𝜃𝑡.
(c)Dynamic Fisher Weight Guard. Guided by the L𝐶𝐾𝑇 andL𝐶𝑀𝐶 losses, we employ Fisher Importance Mapping (FIM) to
dynamically assess weight importance, resulting in an adaptively enhanced L2 regularization loss,L 𝐷𝐹𝐺 .
•We propose MGTIL, a comprehensive continual learning benchmark for the medical general domain, covering
numerous datasets across diverse fields. It supports three evaluation scenarios: intra-domain transfer, inter-domain
transfer, and high-difficulty task retention.
•We propose PRIMED, which dynamically adjusts the reference-data distribution and knowledge ratio to predict model-
weight importance in real time for complex medical continual learning, preserving fine-grained intra-modal features
and bridging significant inter-modal domain gaps. It achieves state-of-the-art performance across all evaluations on
MGTIL.
2 Related Work
Medical Continual Learning.Continual learning (CL) is an extensively researched topic in machine learning and
computer vision, encompassing various scenarios like class-incremental, task-incremental, and domain-incremental
learning. Strategies to address this challenge are generally categorized as regularization-based [ 2,41,48,75],
distillation-based [ 10,26,41,61,76,85,90], architecture-based [ 24,43,58,69], or rehearsal-based [ 10,12,78]. In
recent years, several studies have explored the application of CL in medical image analysis, including works on MRI
segmentation and classification [ 47,57,93], X-ray analysis [ 8,67], pathology image analysis [ 28,35,44], skin image
classification [ 7] and robotic surgery [ 16,25]. In response to the escalating size of models and their corresponding
data needs, regularization-based [ 52] and LoRA-based [ 83,84,87] CL strategies are playing a crucial role in the
effective pre-training of medical foundation models. To date, the literature notably lacks a systematic methodology
and a corresponding benchmark to investigate the CL performance of medical Vision Language Models (VLMs) and
concurrently optimize their task-incremental and domain-incremental capabilities.
Continual Learning for VLMs.VLMs exhibit strong generalization and zero-shot capabilities [ 11,36,48]. While
extensive research has focused on finetuning general-domain VLMs [ 21,33,81] to boost downstream performance while
retaining generalization, finetuning for complex medical scenarios remains underexplored. Parameter-efficient methods
like prompt tuning [ 37,69,91] and adapters [ 43,83,84] are often suboptimal for complex tasks due to limited learnable
parameters, prompting exploration into robust backbone finetuning to balance stability and plasticity. In CL scenarios
3

with domain shifts, ZSCL [90] preserves zero-shot capabilities via distillation from external datasets, which SND [85]
enhances using multi-teacher distillation and GIFT [ 76] optimizes by replacing the dataset with a generative model to
reduce storage. Our approach, however, employs a multi-level, multi-modal Retrieval-Augmented Generation (RAG)
framework to construct a dynamic medical external dataset, thereby enhancing the granularity and dynamism of replayed
content.
Generalist Medical Foundation Models.As medical expert language models [ 29,45,77] have advanced, large-scale
image–text pretrained VLMs have emerged in the medical field. Contrastive language–vision pretraining aligns images
and text within a unified embedding space, delivering strong performance in visual tasks like classification and prognosis.
Among generalist models, PMC-CLIP [ 50] leverages one million PMC-OA image–caption pairs, trained with dual
contrastive learning and masked language modeling. BiomedCLIP [ 88] expands this to PMC-15M pairs and adopts a
ViT [ 23] visual encoder. UniMed-CLIP [ 39] extends to more diverse tasks, including generative modeling. Ye et al. [ 80]
introduced continual learning across medical domains to improve efficiency and preserve data privacy. Lozano et al. [ 52]
used DINOv2 [ 55] to recluster PUBMED data, forming the BIOMEDICA-22M dataset and applying weighted continual
pretraining. Recently, MMKD-CLIP [ 71] combined strengths of prior medical contrastive models via multi-teacher
knowledge distillation, achieving superior overall results.
3 Methodology
3.1 Preliminaries
Continual Learning.Given a sequence [T1,T2,···,T𝑛]of𝑛tasks, continual training is performed sequentially on
each taskT𝑖=(D𝑖,𝐶𝑖), where𝑖=1,...,𝑛 . Here,D𝑖denotes the dataset of task 𝑖, represented as{𝒙𝑖
𝑗,𝒚𝑖
𝑗}𝑁𝑖
𝑗=1, where
𝒙𝑖
𝑗is an image, 𝒚𝑖
𝑗is a one-hot vector representing its ground truth label, and 𝑁𝑖is the total number of images in the
dataset. The class names 𝐶𝑖={𝑐𝑖
𝑗}𝑚𝑖
𝑗=1map each label to its corresponding object name, where 𝑚𝑖denotes the number
of classes in taskT𝑖. The goal of continual training is to maintain strong performance across all tasks.
We focus on task-incremental and domain-incremental learning in medical continual learning (CL). During inference,
the image 𝒙is given with its task identity 𝑡, so the model only distinguishes classes within 𝐶𝑡. The domain shift is
induced by the task order defined in the benchmark.
Medical CLIP.The CLIP [ 59] model includes an image encoder 𝑓𝑖𝑚𝑎𝑔𝑒 and a text encoder 𝑓𝑡𝑒𝑥𝑡. For image classification,
each class𝑐in taskT𝑖is converted into a sentence like “a photo of 𝑐”. Then𝑓𝑡𝑒𝑥𝑡encodes them into text embeddings
𝒕𝑖
𝑗𝑚𝑖
𝑗=1. The image encoder encodes an input image 𝒙𝑘. The cosine similarity between the image and text embeddings is
𝒔𝑖
𝑘,𝑗=⟨𝒕𝑖
𝑗, 𝑓𝑖𝑚𝑎𝑔𝑒(𝒙𝑘)⟩. The class with the highest score is taken as the prediction. The CLIP architecture typically
uses classification losses such as Cross-Entropy ( L𝐶𝐸) or Binary Cross-Entropy ( L𝐵𝐶𝐸) loss to perform fine-tuning on
downstream tasks.
Although Medical CLIP slightly differs from the original CLIP in its training corpus [ 29,45] and architecture [ 17,68,82],
it retains the same zero-shot and fine-tuning protocol. Following ZSCL’s model selection strategy [ 90], we adopt
ViT-B/16 [ 23] for experiments. To maintain architectural consistency, strong performance, and prevent data leakage, we
use BiomedCLIP [ 88] as the backbone.Appendix F.1andF.2contains the rationale for our baseline model selection
and demonstrates that our method achieves superior performance on other backbones as well.
Offline Multimodal Retrieval Database.Rapid advancements in Generalist Medical Foundation Models [ 4,5,50,52,88]
have refined PubMed multimodal data collection. Following BIOMEDICA [ 52], we gathered and processed all literature
data, performing segmentation and pseudo-labeling of images, captions, and text. However, much data is unrelated
to medical imaging or uses multi-image formats, which are suboptimal for CL. To enhance retrieval efficiency, we
compressed textual information and applied a pipeline to decompose multi-image entries, constructing an 18M-scale
retrieval database with Qwen3-Embedding-8B [ 89]. Database construction details are in theAppendix D.1. We also
curated a 3,000-question pool from various sources (literature, datasets, QA reformulation, Wikipedia), refined by
domains and anatomical sites.
4

3.2 Dynamic Multi-stage Retrieval Mechanism
Dense Retrieval via Embedding.Based on the definitions from Sec. 3.1, we use the retrieval database 𝑆and the medical
question pool 𝑀0.𝑆contains tuples(ˆs,c,i) , where ˆsis an embedding vector, cthe caption, and ithe image. For CL over
𝑛tasks[T1,...,T𝑛], we dynamically add previous task labels. Before taskT𝑖’s training, the pool𝑀𝑖is:
𝑀𝑖=𝑀0∪ 𝑖−1Ø
𝑘=1𝐷𝑘!
.(1)
For each query 𝑚∈𝑀𝑖, let𝑆𝑒={ˆs|(ˆs,c,i)∈𝑆} be the set of all normalized vectors from the database. We compute
its cosine similarity with all ˆs∈𝑆𝑒. The query𝑚is first embedded by modelΦ 𝐸[89] and then normalized:
⟨𝑚,ˆs⟩=Φ𝐸(𝑚)
∥Φ𝐸(𝑚)∥ 2·ˆs,∀ ˆs∈𝑆𝑒,(2)
U𝑖
𝑚={⟨𝑚, ˆs⟩|ˆs∈𝑆𝑒}.(3)
To ensure𝐶𝑖
𝑚is robust to numerous tied scores, we avoid simple top-𝑘selection, instead defining a dynamic threshold
𝜏𝑚using the𝑘′-th largest unique similarity score:
𝑘′=min(𝑘,|U𝑖
𝑚|), 𝜏𝑚=(sort↓(U𝑖
𝑚))[𝑘′].(4,5)
Here,𝑘is the target retrieval size. The final candidate set𝐶𝑖
𝑚includes all entries scoring≥𝜏 𝑚, for reranking:
𝐶𝑖
𝑚={(ˆs,c,i)∈𝑆|⟨𝑚, ˆs⟩≥𝜏𝑚}.(6)
Rerank and Gate.To refine the initial candidate set 𝐶𝑖
𝑚, we first rerank all candidates using a VLM [ 88] encoder (Φ𝑉)
to compute a cross-modal score 𝑆𝑣. We then mitigate redundancy by grouping by caption and retaining only the top 𝑘𝑣
images per group, creating an intermediate set 𝐶𝑖∗
𝑚. This set is processed by a lexical gate, which uses BM25 [ 63] to
filter the set down to 𝑘candidates if its size 𝐶𝑖∗
𝑚exceeds the target 𝑘. Let𝑅𝑡𝑜𝑝𝑘(𝐶,𝑆,𝑘) be the top-𝑘selection operator.
The final set𝐶is:
𝐶𝑖
𝑟=(
𝐶𝑖∗
𝑚 if|𝐶𝑖∗
𝑚|≤𝑘
𝑅𝑡𝑜𝑝𝑘(𝐶𝑖∗
𝑚,𝑆𝑏𝑚25,𝑘)if|𝐶𝑖∗
𝑚|>𝑘.(7)
Finally, the𝑘candidates in𝐶𝑖
𝑟are sorted by𝑆 𝑣.
Dynamic Siphon.To enhance knowledge retention, we divide the 𝑀𝑖into three mutually exclusive subsets: 𝑀task,
𝑀domain , and𝑀gen, which correspond to task-specific, domain-related, and general-purpose queries, respectively. The
formal definition is given as follows:
𝑀𝑡𝑎𝑠𝑘=𝑖−1Ø
𝑘=1𝐷𝑘, 𝐶𝑡𝑎𝑠𝑘=𝑖−1Ø
𝑘=1{Φ𝐷(𝑥)|𝑥∈𝐷𝑘},(8,9)
𝑀𝑑𝑜𝑚𝑎𝑖𝑛={𝑚∈𝑀0|Φ𝐷(𝑚)∈𝐶𝑡𝑎𝑠𝑘},(10)
𝑀𝑔𝑒𝑛=S(𝑀0\𝑀𝑑𝑜𝑚𝑎𝑖𝑛,𝑛).(11)
LetΦ𝐷[88] map𝑥to its domain category, 𝐶taskbe the set of 𝑖−1 previous task domains, Sbe a random sampler, and
𝑛be the number of samples. The final tuple set 𝐶𝑖is generated via top- 𝑘sampling with distinct parameters 𝑎,𝑏,𝑐 at
different levels.
𝐶𝑖= 
𝑅𝑡𝑜𝑝𝑘(𝐶𝑖
𝑟,𝑆𝑣,𝑎) ∀𝑚∈𝑀 𝑡𝑎𝑠𝑘
𝑅𝑡𝑜𝑝𝑘(𝐶𝑖
𝑟,𝑆𝑣,𝑏) ∀𝑚∈𝑀 𝑑𝑜𝑚𝑎𝑖𝑛
𝑅𝑡𝑜𝑝𝑘(𝐶𝑖
𝑟,𝑆𝑣,𝑐) ∀𝑚∈𝑀 𝑔𝑒𝑛(12)
5

3.3 Distillation, Alignment and Guard
Contrastive Knowledge Transfer.For a batch of 𝐵image-text pairs from 𝐶𝑖, we use the BiomedCLIP [ 88] image
encoder𝑓𝑖
𝑖𝑚𝑎𝑔𝑒and text encoder𝑓𝑖
𝑡𝑒𝑥𝑡to compute the𝐵×𝐵cross-modal similarity matrix𝑀𝑖=(m𝑖
𝑘,𝑗)as:
m𝑖
𝑘,𝑗=⟨𝑓𝑖
𝑡𝑒𝑥𝑡(c𝑖
𝑘,𝑗), 𝑓𝑖
𝑖𝑚𝑎𝑔𝑒(i𝑖
𝑘,𝑗)⟩.(13)
Then, the teacher model from the task T𝑖−1, with encoders 𝑓𝑖−1
𝑖𝑚𝑎𝑔𝑒and𝑓𝑖−1
𝑡𝑒𝑥𝑡, processes the same batch to get its similarity
matrix𝑀𝑖−1. Logits are similarities scaled by𝜏:
𝑍𝑖=𝑀𝑖
𝜏, 𝑍𝑖−1=𝑀𝑖−1
𝜏.(14)
To compute image-to-text and text-to-image alignments, we convert similarity scores to probability distributions using 𝜎
and measure theirKLdivergence.
L𝐷𝑖𝑠𝑡𝑖𝑙𝑙−𝑖2𝑡=𝐵∑︁
𝑖=1KL
𝜎(𝑍𝑖−1
𝑘,:)∥𝜎(𝑍𝑖
𝑘,:)
,(15)
L𝐷𝑖𝑠𝑡𝑖𝑙𝑙−𝑡2𝑖=𝐵∑︁
𝑗=1KL
𝜎(𝑍𝑖−1
:,𝑗)∥𝜎(𝑍𝑖
:,𝑗)
.(16)
The final Contrastive Knowledge Transfer lossL 𝐶𝐾𝑇 is:
L𝐶𝐾𝑇=L𝐷𝑖𝑠𝑡𝑖𝑙𝑙−𝑖2𝑡+L𝐷𝑖𝑠𝑡𝑖𝑙𝑙−𝑡2𝑖.(17)
Cross-Modality Consistency.In CL knowledge distillation, the teacher model (one task behind the student) also suffers
from catastrophic forgetting, notably decoupling cross-modal knowledge. We thus perform contrastive learning with 𝑀𝑖
against the identity matrix for𝐶𝑖to get Cross-Modality Consistency lossL 𝐶𝑀𝐶 :
L𝐶𝑀𝐶=L𝐴𝑙𝑖𝑔𝑛−𝑖2𝑡+L𝐴𝑙𝑖𝑔𝑛−𝑡2𝑖.(18)
Combining the Cross-Entropy ( L𝐶𝐸) classification loss for task T𝑖, the total training loss for the model is defined as:
L𝑇𝑟𝑎𝑖𝑛=L𝐶𝐸+𝛼L𝐶𝐾𝑇+𝛽L𝐶𝑀𝐶,(19)
where𝛼and𝛽are trade-off hyperparameters.
Dynamic Fisher Weight Guard.Applying a MSE [ 18] penalty between the weights of the fine-tuned VLM and its
original pre-trained state [ 86] can effectively mitigate overfitting induced by the cross-entropy gradient and address the
catastrophic forgetting problem in the foundational model. Elastic Weight Consolidation (EWC) [ 40] is a typical weight
consolidation method that imposes a parameter importance-weighted𝑙 2loss [32], as follows:
L𝐸𝑊𝐶=∑︁
𝑖W𝜃𝑡−1
𝑖·
𝜃𝑡
𝑖−𝜃𝑡−1
𝑖2
.(20)
Parameter importance W𝜃𝑡−1
𝑖is the Fisher Information Matrix diagonal. We further propose: if fine-tuning interference
causes forgetting, shouldn’t L𝐶𝐸also dynamically adapt to distillation and alignment shifts? The Fisher Information
Matrix evolving via gradient backpropagation is a natural mechanism for this:
𝑊(𝑗)
𝑖=©­­
«𝜕
L(𝑗)
Train−L(𝑗)
CE
𝜕𝜃𝑡
𝑖ª®®
¬2
,(21)
L(𝑗)
DFG=∑︁
𝑖𝑊(𝑗)
𝑖·
𝜃𝑡(𝑗)
𝑖−𝜃𝑡−1
𝑖2
.(22)
whereW(𝑗)
𝜃𝑡
𝑖denotes the diagonal Fisher information of model parameter𝜃𝑡
𝑖at the𝑗𝑡ℎoptimization step.
3.4 MGTIL Benchmark
As shown in Fig. 3(b), MGTIL features two distinct tasks to evaluate the CL performance of VLMs at different levels.
6

Table 1Comparison of SOTA methods on HieraMedTransfer Order I and II. Gray Background indicates baseline.
Red Background&boldindicate best results.Red/bluefonts indicateincrease/decreaserelative to baseline.
Method PublicationHieraMedTransfer Order I HieraMedTransfer Order II
TransferΔ Avg.Δ LastΔ TransferΔ Avg.Δ LastΔ
Zero-shot - 57.5 - 52.9 - 53.0 - 57.5 - 52.9 - 53.0 -
Continual FT - 51.4 - 67.6 - 70.8 - 47.7 - 61.9 - 58.6 -
𝑙2baseline - 53.6 0.0 68.5 0.0 74.1 0.0 45.8 0.0 63.9 0.0 66.8 0.0
LwF [48] TPAMI 2017 43.7-9.9 54.3-14.2 65.1-9.0 47.0+1.2 61.7-2.2 61.7-5.1
iCaRL [61] CVPR 2017 52.8-0.8 70.7+2.2 77.8+3.7 48.1+2.3 66.0+2.1 73.9+7.1
WiSE-FT [ 75] CVPR 2022 53.2-0.4 69.4+0.9 75.2+1.1 48.3+2.5 64.8+0.9 65.9-0.9
ZSCL [90] ICCV 2023 57.7+4.1 70.5+2.0 77.6+3.5 45.2-0.6 65.0+1.1 76.9+10.1
MoE-CL [83] CVPR 2024 56.8+3.2 70.7+2.2 76.1+2.0 47.6+1.8 66.1+2.2 74.2+7.4
SND [85] ECCV 2024 52.0-1.6 67.5-1.0 70.1-4.0 46.9+1.1 61.0-2.9 53.3-13.5
DIKI [69] ECCV 2024 56.3+2.7 70.9+2.4 77.4+3.3 46.6+0.8 65.7+1.8 76.7+9.9
GIFT [76] CVPR 2025 53.4-0.2 69.8+1.3 75.2+1.1 46.8+1.0 65.0+1.1 71.3+4.5
PRIMED𝑢𝑛𝑖 - 57.1+3.5 72.7+4.2 81.7+7.6 48.0+2.2 67.7+3.8 77.0+10.2
PRIMED𝑑𝑦𝑛 - 58.3 +4.7 73.1 +4.6 82.1 +8.0 48.5 +2.7 68.0 +4.1 81.2 +14.4
avg
avg
avg
 Tn  Tn-1        T4     T3    T2    T1Task Training StepT1     T2      T3     T4         Tn-1     Tn
Transfer Last Avg.   (a)                                                         (b)         

Intra-Domain
Gap

Cross-Domain
Gap
X-ray Domain
Patho DomainEvaluation Task ID                                  HieraMedTransfer           
MedXtreme
VLM
Figure 3(a) Illustration of calculating CL metrics: Transfer, Avg.,
and Last. (b) Overview of MGTIL’s two tasks, evaluating CL
scenarios across intra-domain, cross-domain, and task retention.HieraMedTransfer Benchmark.Intra-domain med-
ical images are highly similar [ 6,13,31,62], requiring
continual discrimination to memorize fine-grained fea-
tures. Concurrently, retaining memory across large
cross-domain gaps is a severe challenge [ 73]. We pro-
pose HieraMedTransfer to address both challenges. We
curated nine datasets [ 1,3,9,19,30,34,64,65,92]
from three domains (X-ray, pathology, and fundus) cov-
ering diverse resolutions, regions, and involvement. We
designed two sequences: OrderI trains sequentially by
domain to simulate pre-training, while OrderII uses a
randomized alphabetical task arrangement to simulate
a realistic clinical CL scenario.
MedXtreme Benchmark.MedXtreme targets the chal-
lenging tasks [ 38,42,53,60,66,72] across 6 medical
domains: X-ray, fundus, pathology, endoscopy, dermoscopy and cell, each involving up to 33 classes. These domains
and tasks exhibit insufficient training data and limited exploration. Experiments were conducted with alphabetical
(OrderI) and random (OrderII) task sequences. Detailed benchmark datasets are presented in theAppendix D.2.
Evaluation Metrics.We perform the multi-task evaluation following the ZSCL, as depicted in Fig. 3(a). Zero-shot
transfer ability enables predictions on all datasets. The Avg metric is the average accuracy across all datasets and
timestamps. The Last metric is the average performance of all tasks after CL. The Transfer metric is the average
task performance in the upper-right triangle of the matrix, measuring the preservation of zero-shot transfer ability.
Tasks are first averaged to ensure equal dataset weighting. Before learning task 𝑖, tasks𝑗≥𝑖 are not fine-tuned; their
performance thus indicates zero-shot ability. However, even Medical VLMs exhibit insufficient zero-shot performance
on the MedXtreme. Therefore, we adopt traditional Task-CL metrics, such as ACC, AUC, BWT [ 22] and Forgetting [ 14].
4 Experiments
4.1 Experimental Setting
Implementation Details.We conduct all experiments on a 4-NVIDIA A6000 GPU workstation, using BiomedCLIP [ 88]
as the backbone. Each MGTIL task is trained for 1,000 iterations with a batch size of 64 and a 1×10−5learning rate.
Full retrieval and model training settings and hyperparameters are detailed in Tab. 4 and theAppendix E.
7

Table 2Comparison of SOTA methods on MedXtreme Order I and II. Gray Background indicates baseline. Red Background
&boldindicate best results.Red/bluefonts indicateincrease/decreaserelative to baseline.
Method PublicationMedXtreme Order I MedXtreme Order II
ACCΔ AUCΔ BWTΔ ACCΔ AUCΔ BWTΔ
Zero-shot - 9.9 - 61.2 - - - 9.9 - 61.2 - - -
Continual FT - 61.0 - 86.7 - -15.1 - 60.0 - 84.1 - -16.0 -
𝑙2baseline - 61.1 0.0 82.9 0.0 -10.0 0.0 57.3 0.0 81.2 0.0 -14.5 0.0
LwF [48] TPAMI 2017 51.5-9.6 81.4-1.5 -8.3+1.7 44.2-13.1 81.0-0.2 -11.3+3.2
iCaRL [61] CVPR 2017 65.9+4.8 84.8+1.9 -3.1+6.9 64.2+6.9 84.3+3.1 -5.1+9.4
WiSE-FT [ 75] CVPR 2022 65.1+4.0 86.5+3.6 -8.5+1.5 64.1+6.8 85.6+4.4 -9.2+5.3
ZSCL [90] ICCV 2023 53.7-7.4 79.9-3.0 -6.5+3.5 48.3-9.0 78.6-2.6 -13.7+0.8
MoE-CL [83] CVPR 2024 65.3+4.2 84.9+2.0 -4.1+5.9 64.7+7.4 84.5+3.3 -4.4+10.1
SND [85] ECCV 2024 61.7+0.6 85.3+2.4 -13.9-3.9 57.6+0.3 84.3+3.1 -18.8-4.3
DIKI [69] ECCV 2024 64.8+3.7 86.7+3.8 -9.2+0.8 63.1+5.8 85.2+4.0 -10.3+4.2
GIFT [76] CVPR 2025 66.0+4.9 86.6+3.7 -3.7+6.3 65.7+8.4 85.2+4.0 -4.1+10.4
PRIMED𝑢𝑛𝑖 - 66.2+5.1 86.7+3.8 -4.4+5.6 64.5+7.2 85.4+4.2 -6.6+7.9
PRIMED𝑑𝑦𝑛 - 68.6 +7.5 87.4 +4.5 -2.7 +7.3 68.1 +10.8 86.3 +5.1 -3.4 +11.1
Datasets and Task Sequence.Following Sec. 3.4’s design, we detail the datasets and order. HieraMedTransfer is
evaluated on RANZCR [ 64], CheXchoNet [ 9], PD [ 3], Breakhis [ 65], Chaoyang [ 92], NuCLS [ 34], Eyepacs [ 30],
AIROGS [ 19], and FARFUM-RoP [ 1]. The two task orders are: Order I (as introduced above) and Order II (alphabetical).
MedXtreme comprises AOD [ 60], NCT100K [ 38], PITVIS [ 66], ISIC2024 [ 42], NIH-Chest-Xray [ 72], and BMC [ 53].
Similarly, the task orders are Order I (alphabetical) and Order II (random, as introduced above). TheAppendix D.2
details the dataset composition, distribution, and volume, confirming its isolation to prevent data leakage.
Reference Dataset Construction.Distillation-based methods [ 85,90] require a reference dataset. Since prior work
focused on natural images, we adapted these methods to the medical domain for a fair comparison. While existing
methods use uniform sampling from coarse-grained labels, we construct an equivalently sized reference dataset to
ZSCL ’s via proportional retrieval using 30 medical domain keywords. Text-to-image methods [ 76] generate data from
customized labels; we utilize our fine-grained question pool as generation prompts. Notably, while our method expectedly
excels in dynamic multi-stage retrieval, it also outperforms all reference-set distillation methods under the uniform
retrieval setting showing in Tab. 1 and Tab. 2.
4.2 Comparison with State-of-the-art Methods
The average performance of different methods on Sequence I and Sequence II of the MGTIL benchmark is presented in
Tab. 1 and Tab. 2, respectively (more granular and comprehensive numerical results are available in theAppendix D.2).
At the top of the table, the Zero-shot method is used to obtain the logical upper bound for transfer, while continual FT
without any strategy provides the logical lower bounds for Avg. and Last. 𝑙2regularization, which effectively constrains
the magnitude of fine-tuning, yields a more balanced result and is treated as our baseline.
80
70
60
50
40
1 2 3 4 5 6 1 2 3 4 5 680
75
70
65
60
55
50Accuracy
Accuracy
Task Sequence Task SequenceAcc. of 1st task (AOD) in MedXtreme Order I and Order II
Continue-FT
WISE-FT
ZSCL
SND
GIFT
PRIMEDContinue-FT
WISE-FT
ZSCL
SND
GIFT
PRIMED
Figure 4ACC of 1st task (AOD) in MedXtreme Order I and IIResults on HieraMedTransfer Benchmark.Order I
uses sequential fine-tuning on uniform domains, creat-
ing a regular sequence. Thus, ZSCL [ 90] with reference
data and sequential MOE-CL [ 83] adapt better here.
Conversely, Order II randomizes inter-task domain dis-
tance. This greater challenge explains its lower average
performance versus Order I. Here, rehearsal, reference
data, and prompt tuning methods [ 69] performed well.
PRIMED𝑢𝑛𝑖using ZSCL ’s reference data beat all com-
parators, showing our loss function’s suitability for
complex medical tasks. Moreover, PRIMED𝑑𝑦𝑛using
dynamic multi-stage retrieval achieved superior results. It reached state-of-the-art (SOTA) on all metrics with up to a
5.6% improvement and avoids data contamination from rehearsal or generative methods.
8

Table 4 Ablation experiments.Our method uses Last CLIP and Hierarchical Retrieval for reference data, applies simultaneous
image-text distillation, and leverages the DFG algorithm for dynamic weighting. Default settings are marked in Red Background .
(a)Teacher Model.
Teacher Transfer Avg. Last
Initial CLIP 57.9 70.7 78.2
Last CLIP 58.3 73.1 82.1
WISE(0.5) 58.1 71.4 79.1(b)Distillation Loss.
Loss Transfer Avg. Last
Image-only 57.7 72.4 81.3
Text-only 58.2 73.0 80.5
Contrastive 58.3 73.1 82.1(c)Scale of Distillation.
CKT Scale Transfer Avg. Last
𝛼=0.5 58.3 72.8 81.3
𝛼=1 58.3 73.1 82.1
𝛼=1.5 58.3 73.0 81.9
(d)Scale of Image-Text Alignment.
CMC Scale Transfer Avg. Last
𝛽=0.0 57.2 72.8 82.5
𝛽=0.25 58.3 73.1 82.1
𝛽=0.5 58.1 72.5 81.4(e)Regularization Term.
Method Transfer Avg. Last
𝑙2 57.2 72.9 81.8
EWC 57.2 71.3 80.6
DFG 58.3 73.1 82.1(f)Retrieval Method.
Method Transfer Avg. Last
BM25 55.1 72.2 81.8
Embedding 56.1 72.5 81.5
Hierarchical 58.3 73.1 82.1
Results on MedXtreme Benchmark.SOTA performance on both task sequences is achieved by PRIMED𝑑𝑦𝑛. As peak
performance typically occurs post-fine-tuning, BWT and Forgetting exhibit an inverse relationship. We therefore report
BWT scores exclusively. Tab. 2 demonstrates that dynamic multi-stage retrieval is essential for constructing the granular
memory required to master complex tasks. Consequently, replay-based [ 61] and generative-based [ 76] methods, while
competitive, are outperformed by our approach. Crucially, beyond its quantitative superiority, PRIMED𝑑𝑦𝑛obviates the
need for original training data, positioning it as a more practical method for real-world medical scenarios. Fig. 4 shows
the first task (AOD) ACC variation during training. Our method avoids abrupt catastrophic forgetting and exhibits the
lowest overall forgetting.
4.3 Ablation Study
Our method is validated on all MGTIL sequences. Tab.3 and Tab.4 provide ablations for HieraMedTransfer Order I. Full
results are in theAppendix E.
Modular Level Analysis.As shown in Tab.3, we decouple the three modules via distinct losses and conduct a
comprehensive ablation. Contrastive Knowledge Transfer(CKT)is the foundational component; its removal causes a
drastic performance degradation across all levels. Cross-Modality Consistency(CMC)leverages mini-batch contrastive
learning to preserve zero-shot capabilities, preventing excessive knowledge distribution shifts. Finally, the Dynamic
Fisher Weight Guard(DFG)provides dynamic knowledge refinement, comprehensively improving overall performance.
Table 3Ablation study of different modules.
+CKT +CMC +DFG Transfer Avg. Last
✓ 56.2 71.8 82.6
✓ 54.3 68.9 78.9
✓ ✓ 56.9 71.6 82.2
✓ ✓ 57.2 72.8 82.5
✓ ✓ 56.7 70.2 77.1
✓ ✓ ✓ 58.3 73.1 82.1Component and Hyperparameter Analysis.As
shown in Tab. 4, we conduct an ablation study to val-
idate the feasibility and necessity of the components
and hyperparameters used in our training process. The
best-performing results are demonstrated on the Hier-
aMedTransfer Order I. To demonstrate the robustness of
our method, all hyperparameters in the aforementioned
experiments are uniformly set, using a random seed of
42. The settings are divided into 4 main categories.
For the teacher model, we compare three variants: the
initial model, the model from the previous fine-tuning
task (Last), and an ensemble model based on the WISE [ 75] weight-averaging strategy. It is demonstrated that for
medical VLMs, the ’Last’ model achieves the best performance, likely due to discrepancies in knowledge distribution.
This finding also holds true for the highly challenging data scenarios represented by MedXtreme. Regarding the training
loss hyperparameters, we utilized the settings that achieved peak performance. At the regularization constraint level, we
demonstrate the value of our dynamic approach, which outperforms standard 𝑙2and EWC. In terms of dynamic retrieval,
our dynamic multi-stage retrieval shows significant performance gains. Furthermore, we present an analysis in the
Appendix E.4on the impact of the optimal class-wise retrieval ratio and the total volume of retrieved data within our
Dynamic Siphon module.
9

A Fundus photo of 
Plus Retinopathy of Prematurity
Caption     
Color fundus image ... of 
an avascular peripheral 
retina at 35 weeks of 
postmenstrual age ...Image    Ours:TOP1
Caption     
Pre injection fundus 
photograph of the right 
eye of a baby with 
threshold ROPImage    
Caption     
Fundus photograph 
showing fresh ROP 
laser marks ... at 1 
year of ageImage    Caption     
Fundus photos show 
persistent 
neovascularisations at 
optic discs  
Caption     
...of Purtscher-Like 
Retinopathy in a 29-
Year-Old Female. A 
photo of her skin. Image    Image    
A pathology photo 
of serrated lesion in the colonCaption     
Traditional serrated 
adenoma of colonic 
mucosa (sigmoid) 
(H&E staining, 2×).Image    Ours:TOP1
Caption     
Histologic appearance 
of the serrated lesion 
in the sigmoid colon-
hyperplastic polyp Image    
Caption     
The pathological 
findings of sessile 
serrated lesion are 
shownImage    Caption     
Serrated adenoma in 
the ascending colon.
Caption     
...a Effect ... in 
visibility. A nonpolypoid 
lesion of ... in the 
ascending colon ...Image    Image    
Ours:TOP2
W/O Rerank and GateOurs:TOP3
BM25
Ours:TOP2 Ours:TOP3
W/O Rerank and Gate BM25
Figure 5In our qualitative visualization for retrieval using two prompts, we showcase theTop-3 resultsof our proposed method,
comparing them against anablation variantwithout the Rerank and Gate way and the pureBM25baseline. Our approach offers two
significant advantages: 1. It avoids spurious matches that focus merely on keywords while failing to capture the correct domain context.
2. It is capable of retrieving more fine-grained matches related to lesion characteristics and severity, moving beyond superficial textual
similarity.
4.4 Retrieval Visualization
Our method demonstrates two significant retrieval advantages. First, it mitigates spurious textual correlations, preventing
false pairings. For example, a naive keyword-based method might erroneously pair pathology and endoscopy images via
the term“colon”—a fundamental domain mismatch that precludes effective learning. Second, our approach transcends
simple keyword matching. Although“Retinopathy of Prematurity”is a pediatric condition, the term lacks explicit
pediatric keywords. Our method nonetheless retrieves the semantically correct match, proving robust in the absence of
direct keyword overlap.
5 Conclusion
In this paper, we propose PRIMED, a novel Continual Learning (CL) framework for Generalist Medical Foundation
Models. To facilitate dataset curation, we have developed a new retrieval library and question pool. A complete pipeline
encompassing retrieval, distillation, alignment, and guarding is dynamically implemented by PRIMED, enhancing
its adaptability to the true data distribution of medical imaging and the realities of clinical practice. Comprehensive
experiments demonstrate that our method achieves optimal performance in domain spanning, micro-level feature capture,
and hard task retention. Methodologically, it explores the relationship between retrieval and distillation as two distinct
forms of model memory. Clinically, this method can be utilized for the real-time updating of medical diagnostic
systems. From a data ethics perspective, we have audited the relevant data for ethical and copyright compliance, and will
release all associated content upon secondary confirmation. We provide a further discussion on limitations and ethical
considerations in theAppendix.
10

References
[1]Morteza Akbari, Hamid-Reza Pourreza, Elias Khalili Pour, Afsar Dastjani Farahani, Fatemeh Bazvand, Nazanin Ebrahimiadib,
Marjan Imani Fooladi, and Fereshteh Ramazani K. Farfum-rop, a dataset for computer-aided detection of retinopathy of
prematurity.Scientific Data, 11(1):1176, 2024.
[2]Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. Memory aware synapses:
Learning what (not) to forget. InProceedings of the European conference on computer vision (ECCV), pages 139–154, 2018.
[3] A Asraf and Z Islam. Covid19, pneumonia and normal chest x-ray pa dataset. mendeley data v1 (2021), 2021.
[4]Negin Baghbanzadeh, Sajad Ashkezari, Elham Dolatabadi, and Arash Afkanpour. Open-pmc-18m: A high-fidelity large scale
medical dataset for multimodal representation learning.arXiv preprint arXiv:2506.02738, 2025.
[5]Negin Baghbanzadeh, Adibvafa Fallahpour, Yasaman Parhizkar, et al. Advancing medical representation learning through
high-quality data. InInternational Conference on Medical Image Computing and Computer-Assisted Intervention, pages 24–33.
Springer, 2025.
[6]Nourhan Bayasi, Ghassan Hamarneh, and Rafeef Garbi. Culprit-prune-net: Efficient continual sequential multi-domain learning
with application to skin lesion classification. InInternational Conference on Medical Image Computing and Computer-Assisted
Intervention, pages 165–175. Springer, 2021.
[7]Nourhan Bayasi, Siyi Du, Ghassan Hamarneh, and Rafeef Garbi. Continual-gen: Continual group ensembling for domain-agnostic
skin lesion classification. InInternational Conference on Medical Image Computing and Computer-Assisted Intervention, pages
3–13. Springer, 2023.
[8]Nourhan Bayasi, Jamil Fayyad, Alceu Bissoto, Ghassan Hamarneh, and Rafeef Garbi. Biaspruner: Debiased continual learning
for medical image classification. InInternational Conference on Medical Image Computing and Computer-Assisted Intervention,
pages 90–101. Springer, 2024.
[9]Shreyas Bhave, Victor Rodriguez, Timothy Poterucha, Simukayi Mutasa, Dwight Aberle, et al. Deep learning to detect left
ventricular structural abnormalities in chest x-rays.European heart journal, 45(22):2002–2012, 2024.
[10] Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual
learning: a strong, simple baseline.Advances in neural information processing systems, 33:15920–15930, 2020.
[11] Francisco M Castro, Manuel J Mar ´ın-Jim ´enez, Nicol ´as Guil, Cordelia Schmid, and Karteek Alahari. End-to-end incremental
learning. InProceedings of the European conference on computer vision (ECCV), pages 233–248, 2018.
[12] Hyuntak Cha, Jaeho Lee, and Jinwoo Shin. Co2l: Contrastive continual learning. InProceedings of the IEEE/CVF International
conference on computer vision, pages 9516–9525, 2021.
[13] Tapabrata Chakraborti, Fergus Gleeson, and Jens Rittscher. Contrastive representations for continual learning of fine-grained
histology images. InInternational Workshop on Machine Learning in Medical Imaging, pages 1–9. Springer, 2021.
[14] Arslan Chaudhry, Puneet K Dokania, Thalaiyasingam Ajanthan, and Philip HS Torr. Riemannian walk for incremental learning:
Understanding forgetting and intransigence. InProceedings of the European conference on computer vision (ECCV), pages
532–547, 2018.
[15] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multi-
functionality, multi-granularity text embeddings through self-knowledge distillation.arXiv preprint arXiv:2402.03216, 2024.
[16] Kexin Chen, Yuyang Du, Tao You, Mobarakol Islam, Ziyu Guo, Yueming Jin, Guangyong Chen, and Pheng-Ann Heng.
Llm-assisted multi-teacher continual learning for visual question answering in robotic surgery. In2024 IEEE International
Conference on Robotics and Automation (ICRA), pages 10772–10778. IEEE, 2024.
[17] Yixiong Chen, Shawn Xu, Andrew Sellergren, Yossi Matias, Avinatan Hassidim, Shravya Shetty, Daniel Golden, Alan L Yuille,
and Lin Yang. Coca-cxr: Co ntrastive ca ptioners learn strong temporal structures for chest x-ray vision-language understanding.
InInternational Conference on Medical Image Computing and Computer-Assisted Intervention, pages 78–88. Springer, 2025.
[18] Davide Chicco, Matthijs J Warrens, and Giuseppe Jurman. The coefficient of determination r-squared is more informative than
smape, mae, mape, mse and rmse in regression analysis evaluation.Peerj computer science, 7:e623, 2021.
[19] Coen de Vente, Koenraad A.S ´anchez Vermeer, and Clara I. Airogs: Artificial intelligence for robust glaucoma screening
challenge.IEEE Transactions on Medical Imaging, 43(1):542–557, 2024. doi: 10.1109/TMI.2023.3313786.
11

[20] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In
2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
[21] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers
for language understanding. InProceedings of the 2019 conference of the North American chapter of the association for
computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.
[22] Natalia D ´ıaz-Rodr ´ıguez, Vincenzo Lomonaco, David Filliat, and Davide Maltoni. Don’t forget, there is more than forgetting:
new metrics for continual learning.arXiv preprint arXiv:1810.13166, 2018.
[23] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale.arXiv preprint
arXiv:2010.11929, 2020.
[24] Arthur Douillard, Alexandre Ram ´e, Guillaume Couairon, and Matthieu Cord. Dytox: Transformers for continual learning with
dynamic token expansion. InProceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
9285–9295, 2022.
[25] Senlin Fang, Yiwen Liu, Chengliang Liu, Jingnan Wang, Yuanzhe Su, Yupo Zhang, Hoiio Kong, Zhengkun Yi, and Xinyu Wu.
Probabilistic spiking neural network for robotic tactile continual learning. In2024 IEEE International Conference on Robotics
and Automation (ICRA), pages 530–536. IEEE, 2024.
[26] Enrico Fini, Victor G Turrisi Da Costa, Xavier Alameda-Pineda, Elisa Ricci, Karteek Alahari, and Julien Mairal. Self-supervised
models are continual learners. InProceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
9621–9630, 2022.
[27] Rui Gao and Weiwei Liu. Ddgr: Continual learning with deep diffusion-based generative replay. InInternational Conference on
Machine Learning, pages 10744–10763. PMLR, 2023.
[28] Jiaxiang Gou, Luping Ji, Pei Liu, and Mao Ye. Queryable prototype multiple instance learning with vision-language models for
incremental whole slide image classification. InProceedings of the AAAI Conference on Artificial Intelligence, volume 39,
pages 3158–3166, 2025.
[29] Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng Gao, and Hoifung
Poon. Domain-specific language model pretraining for biomedical natural language processing.ACM Transactions on Computing
for Healthcare (HEALTH), 3(1):1–23, 2021.
[30] Varun Gulshan, Lily Peng, et al. Development and validation of a deep learning algorithm for detection of diabetic retinopathy
in retinal fundus photographs.jama, 316(22):2402–2410, 2016.
[31] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, et al. The many faces of robustness: A critical
analysis of out-of-distribution generalization. InProceedings of the IEEE/CVF international conference on computer vision,
pages 8340–8349, 2021.
[32] Arthur E Hoerl and Robert W Kennard. Ridge regression: Biased estimation for nonorthogonal problems.Technometrics, 12(1):
55–67, 1970.
[33] Edward J Hu, Yelong Shen, Phillip Wallis, et al. Lora: Low-rank adaptation of large language models.ICLR, 1(2):3, 2022.
[34] Weiming Hu, Chen Li, Xiaoyan Li, et al. Gashissdb: A new gastric histopathology image dataset for computer aided diagnosis
of gastric cancer.Computers in biology and medicine, 142:105207, 2022.
[35] Yanyan Huang, Weiqin Zhao, Shujun Wang, Yu Fu, Yuming Jiang, and Lequan Yu. Conslide: Asynchronous hierarchical
interaction transformer with breakup-reorganize rehearsal for continual whole slide image analysis. InProceedings of the
IEEE/CVF International Conference on Computer Vision, pages 21349–21360, 2023.
[36] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig.
Scaling up visual and vision-language representation learning with noisy text supervision. InInternational conference on
machine learning, pages 4904–4916. PMLR, 2021.
[37] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual
prompt tuning. InEuropean conference on computer vision, pages 709–727. Springer, 2022.
[38] Jakob Nikolas Kather, Niels Halama, and Alexander Marx. 100,000 histological images of human colorectal cancer and healthy
tissue, April 2018. URLhttps://doi.org/10.5281/zenodo.1214456.
12

[39] Muhammad Uzair Khattak, Shahina Kunhimon, Muzammal Naseer, Salman Khan, and Fahad Shahbaz Khan. Unimed-clip:
Towards a unified image-text pretraining paradigm for diverse medical imaging modalities.arXiv preprint arXiv:2412.10372,
2024.
[40] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan,
John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks.
Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.
[41] Richard Kurle, Botond Cseke, Alexej Klushyn, Patrick Van Der Smagt, and Stephan G¨ unnemann. Continual learning with
bayesian neural networks for non-stationary data. InInternational Conference on Learning Representations, 2019.
[42] Nicholas Kurtansky, Veronica Rotemberg, Maura Gillis, Kivanc Kose, Walter Reade, and Ashley Chow. Isic 2024 - skin cancer
detection with 3d-tbp, 2024. URLhttps://kaggle.com/competitions/isic-2024-challenge.
[43] Minh Le, An Nguyen, Huy Nguyen, Trang Nguyen, Trang Pham, Linh Van Ngo, and Nhat Ho. Mixture of experts meets
prompt-based continual learning.Advances in Neural Information Processing Systems, 37:119025–119062, 2024.
[44] Byung Hyun Lee, Wongi Jeong, Woojae Han, Kyoungbun Lee, and Se Young Chun. Continual multiple instance learning with
enhanced localization for histopathological whole slide image analysis.arXiv preprint arXiv:2507.02395, 2025.
[45] Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. Biobert: a
pre-trained biomedical language representation model for biomedical text mining.Bioinformatics, 36(4):1234–1240, 2020.
[46] Patrick Lewis, Ethan Perez, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks.Advances in neural
information processing systems, 33:9459–9474, 2020.
[47] Weilu Li, Yun Zhang, Hao Zhou, Wenhan Yang, Zhi Xie, and Yao He. Clms: Bridging domain gaps in medical imaging
segmentation with source-free continual learning for robust knowledge transfer and adaptation.Medical Image Analysis, 100:
103404, 2025.
[48] Zhizhong Li and Derek Hoiem. Learning without forgetting.IEEE transactions on pattern analysis and machine intelligence,
40(12):2935–2947, 2017.
[49] Lin Lin, Jiefeng Long, Zhihe Wan, Yuchi Wang, Dingkang Yang, Shuang Yang, Yueyang Yao, Xu Chen, Zirui Guo, Shengqiang
Li, et al. Sail-embedding technical report: Omni-modal embedding foundation model.arXiv preprint arXiv:2510.12709, 2025.
[50] Weixiong Lin, Ziheng Zhao, Xiaoman Zhang, Chaoyi Wu, Ya Zhang, Yanfeng Wang, and Weidi Xie. Pmc-clip: Contrastive
language-image pre-training using biomedical documents. InInternational Conference on Medical Image Computing and
Computer-Assisted Intervention, pages 525–536. Springer, 2023.
[51] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. Dab-detr: Dynamic anchor
boxes are better queries for detr.arXiv preprint arXiv:2201.12329, 2022.
[52] Alejandro Lozano, Min Woo Sun, et al. Biomedica: An open biomedical image-caption archive, dataset, and vision-language
models derived from scientific literature. InProceedings of the Computer Vision and Pattern Recognition Conference, pages
19724–19735, 2025.
[53] Christian Matek, Sebastian Krappe, Christian M¨ unzenmayer, Torsten Haferlach, and Carsten Marr. Highly accurate differentiation
of bone marrow cell morphologies using deep neural networks on a large image data set.Blood, The Journal of the American
Society of Hematology, 138(20):1917–1927, 2021.
[54] Zichong Meng, Jie Zhang, Changdi Yang, Zheng Zhan, Pu Zhao, and Yanzhi Wang. Diffclass: Diffusion-based class incremental
learning. InEuropean Conference on Computer Vision, pages 142–159. Springer, 2024.
[55] Maxime Oquab, Timoth ´ee Darcet, Th ´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel
Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision.arXiv
preprint arXiv:2304.07193, 2023.
[56] Tiantian Peng, Yuyang Liu, Shuo Yang, Qiuhe Hong, and YongHong Tian. Gnsp: Gradient null space projection for preserving
cross-modal alignment in vlms continual learning.arXiv preprint arXiv:2507.19839, 2025.
[57] Matthias Perkonigg, Johannes Hofmanninger, Christian J Herold, James A Brink, Oleg Pianykh, Helmut Prosch, and Georg
Langs. Dynamic memory to alleviate catastrophic forgetting in continual learning with medical imaging.Nature communications,
12(1):5678, 2021.
[58] Quang Pham, Chenghao Liu, and Steven Hoi. Dualnet: Continual learning, fast and slow.Advances in Neural Information
Processing Systems, 34:16131–16144, 2021.
13

[59] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, et al. Learning transferable
visual models from natural language supervision. InInternational conference on machine learning, pages 8748–8763. PmLR,
2021.
[60] Mohammad Riadur Rashid, Shayla Sharmin, Tania Khatun, Md Zahid Hasan, and Mohammad Shorif Uddin. Eye Disease
Image Dataset, 2024. URLhttps://data.mendeley.com/datasets/s9bfhswzjb/1.
[61] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and
representation learning. InProceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 2001–2010,
2017.
[62] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In
International conference on machine learning, pages 5389–5400. PMLR, 2019.
[63] Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond.Foundations and Trends®
in Information Retrieval, 3(4):333–389, 2009.
[64] Royal Australian and New Zealand College of Radiologists (RANZCR) and Kaggle. RANZCR CLiP - Catheter
and Line Position Challenge. Kaggle Competition, 2021. URL https://www.kaggle.com/competitions/
ranzcr-clip-catheter-line-classification.
[65] Fabio A Spanhol, Luiz S Oliveira, Caroline Petitjean, and Laurent Heutte. A dataset for breast cancer histopathological image
classification.Ieee transactions on biomedical engineering, 63(7):1455–1462, 2015.
[66] Stefanie Speidel, Lena Maier-Hein, Danail Stoyanov, and Stamatia Giannarou. Endoscopic vision challenge 2023, September
2023. URLhttps://doi.org/10.5281/zenodo.8315050.
[67] Shikhar Srivastava, Mohammad Yaqub, Karthik Nandakumar, Zongyuan Ge, and Dwarikanath Mahapatra. Continual domain
incremental learning for chest x-ray classification in low-resource clinical settings. InMICCAI Workshop on Domain Adaptation
and Representation Transfer, pages 226–238. Springer, 2021.
[68] Min Woo Sun, Alejandro Lozano, Javier Gamazo Tejero, Vishwesh Nath, et al. No tokens wasted: Leveraging long context in
biomedical vision-language models.arXiv preprint arXiv:2510.03978, 2025.
[69] Longxiang Tang, Zhuotao Tian, Kai Li, Chunming He, Hantao Zhou, Hengshuang Zhao, Xiu Li, and Jiaya Jia. Mind the
interference: Retaining pre-trained knowledge in parameter efficient continual learning of vision-language models. InEuropean
conference on computer vision, pages 346–365. Springer, 2024.
[70] Ren Tasai, Guang Li, Ren Togo, Minghui Tang, Takaaki Yoshimura, Hiroyuki Sugimori, Kenji Hirata, Takahiro Ogawa, Kohsuke
Kudo, and Miki Haseyama. Continual self-supervised learning considering medical domain knowledge in chest ct images. In
ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE,
2025.
[71] Shansong Wang, Zhecheng Jin, Mingzhe Hu, Mojtaba Safari, Feng Zhao, Chih-Wei Chang, Richard LJ Qiu, Justin Roper,
David S Yu, and Xiaofeng Yang. Unifying biomedical vision-language expertise: Towards a generalist foundation model via
multi-clip knowledge distillation.arXiv preprint arXiv:2506.22567, 2025.
[72] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M Summers. Chestx-ray8: Hospital-scale
chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In
Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2097–2106, 2017.
[73] Zhenyi Wang, Li Shen, Tiehang Duan, Donglin Zhan, Le Fang, and Mingchen Gao. Learning to learn and remember super long
multi-domain task sequence. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
7982–7992, 2022.
[74] Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, et al. Dualprompt: Complementary prompting for
rehearsal-free continual learning. InEuropean conference on computer vision, pages 631–648. Springer, 2022.
[75] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes,
Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al. Robust fine-tuning of zero-shot models. InProceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 7959–7971, 2022.
[76] Bin Wu, Wuxuan Shi, Jinqiao Wang, and Mang Ye. Synthetic data is an elegant gift for continual vision-language models. In
Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2813–2823, 2025.
14

[77] Chaoyi Wu, Weixiong Lin, Xiaoman Zhang, Ya Zhang, Weidi Xie, and Yanfeng Wang. Pmc-llama: toward building open-source
language models for medicine.Journal of the American Medical Informatics Association, 31(9):1833–1843, 2024.
[78] Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye, Zicheng Liu, Yandong Guo, and Yun Fu. Large scale incremental learning.
InProceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 374–382, 2019.
[79] Shipeng Yan, Jiangwei Xie, and Xuming He. Der: Dynamically expandable representation for class incremental learning. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3014–3023, 2021.
[80] Yiwen Ye, Yutong Xie, Jianpeng Zhang, Ziyang Chen, Qi Wu, and Yong Xia. Continual self-supervised learning: Towards
universal multi-modal medical data representation learning. InProceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 11114–11124, 2024.
[81] Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. How transferable are features in deep neural networks?Advances
in neural information processing systems, 27, 2014.
[82] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners
are image-text foundation models.arXiv preprint arXiv:2205.01917, 2022.
[83] Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Ping Hu, Dong Wang, Huchuan Lu, and You He. Boosting continual learning of
vision-language models via mixture-of-experts adapters. InProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 23219–23230, 2024.
[84] Jiazuo Yu, Zichen Huang, Yunzhi Zhuge, Lu Zhang, Ping Hu, Dong Wang, Huchuan Lu, and You He. Moe-adapters++: Towards
more efficient continual learning of vision-language models via dynamic mixture-of-experts adapters.IEEE Transactions on
Pattern Analysis and Machine Intelligence, 2025.
[85] Yu-Chu Yu, Chi-Pin Huang, Jr-Jen Chen, Kai-Po Chang, Yung-Hsuan Lai, Fu-En Yang, and Yu-Chiang Frank Wang. Select and
distill: Selective dual-teacher knowledge transfer for continual learning on vision-language models. InEuropean Conference on
Computer Vision, pages 219–236. Springer, 2024.
[86] Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. InInternational conference
on machine learning, pages 3987–3995. PMLR, 2017.
[87] Haojie Zhang, Yixiong Liang, Hulin Kuang, Lihui Cen, Zhe Qu, Yigang Cen, Min Zeng, and Shichao Kan. Contrastive
regularization over lora for multimodal biomedical image incremental learning.arXiv preprint arXiv:2508.11673, 2025.
[88] Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, et al. Biomedclip: a multimodal biomedical foundation model pretrained
from fifteen million scientific image-text pairs.arXiv preprint arXiv:2303.00915, 2023.
[89] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu,
Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models.arXiv preprint
arXiv:2506.05176, 2025.
[90] Zangwei Zheng, Mingyuan Ma, Kai Wang, Ziheng Qin, Xiangyu Yue, and Yang You. Preventing zero-shot transfer degradation
in continual learning of vision-language models. InProceedings of the IEEE/CVF international conference on computer vision,
pages 19125–19136, 2023.
[91] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16816–16825, 2022.
[92] Chuang Zhu, Wenkai Chen, Ting Peng, Ying Wang, and Mulan Jin. Hard sample aware noise robust learning for histopathology
image classification.IEEE transactions on medical imaging, 41(4):881–894, 2021.
[93] Zhanshi Zhu, Xinghua Ma, Wei Wang, Suyu Dong, Kuanquan Wang, Lianming Wu, Gongning Luo, Guohua Wang, and Shuo
Li. Boosting knowledge diversity, accuracy, and stability via tri-enhanced distillation for domain continual medical image
segmentation.Medical image analysis, 94:103112, 2024.
15

Appendix
Contents
A Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
B Ethical Statement . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
C Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
D Method and Evaluation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
D.1 Data Cleansing Approaches . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
D.2 Benchamrk Database Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
E Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
E.1 Full Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
E.2 Modular Level Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
E.3 Component and Hyperparameter Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
E.4 Dynamic Retrieval Dataset Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
F Backbone Selection and Generalizability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
F.1 Justification for the BiomedCLIP Backbone . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
F.2 Generalizable SOTA Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
G Retrieval Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
A Limitations
Storage Usage.As with all Retrieval-Augmented Generation (RAG) methodologies, the construction and maintenance
of our multimodal retrieval corpus entail additional computational resources and human effort. Furthermore, dynamic
retrieval inherently imposes computational and temporal overheads. These challenges motivate us to advocate for a
phased implementation of our approach. Given that the initial content of the Question Pool is static, retrieval for this
segment can be pre-computed and reused. Consequently, we restrict real-time retrieval operations solely to questions
added subsequently.
The Trade-off in Data Sources.To strictly prevent data leakage and avoid copyright or ethical disputes, we utilized the
PubMed scientific literature database as our retrieval corpus. This choice ensures the reliability of our experimental
results, the generalizability of the method in real-world deployments, and the overall safety and legality of the model.
However, this imposes a limitation: application-level principles, such as fairness and broad ethical considerations, cannot
be explicitly enforced at the algorithmic level during retrieval. Since the vast majority of scientific captions do not
contain sensitive attributes like gender or race, and inferring such information solely from visual data is challenging,
our continual learning method cannot explicitly target these dimensions. Instead, it primarily ensures the absence of
fundamental safety hazards.
B Ethical Statement
This research strictly adheres to the relevant ethical guidelines for medical AI research.
Data Usage and Patient Privacy.All data used in this study were sourced from publicly available research publications
or scientific datasets. All data were fully anonymized and de-identified by the original providers prior to release and
contain no Protected Health Information (PHI). Our usage strictly complies with the Data Use Agreement (DUA) for
PubMed (consistent with BiomedCLIP [ 88]/BIOMEDICA [ 52]) and adheres to the DUAs of all respective open-source
classification datasets involved.
16

Algorithmic Bias.The performance of our model is dependent upon both the backbone and the continual learning
methodology. The data for both components are sourced from scientific literature available on PubMed. Although it is
generally assumed that a dataset comprising tens of millions of samples provides sufficient data diversity, it cannot be
guaranteed that undiscovered biases (e.g., in demographic representation across race, age, or sex) are not present. These
biases may subsequently be learned and amplified by the model. Future work is required to specifically quantify and
mitigate such biases.
C Future Work
We will focus on the real-world rollout of PRIMED and making it easier to deploy.
Data Services.Acknowledging the difficulties associated with large-scale data management, we will release our
retrieval database subject to a secondary ethical audit. We will also establish a cloud service where users can retrieve
information by merely uploading questions. Additionally, we support data streaming to streamline local deployment and
the augmentation of proprietary retrieval repositories.
Rare Disease Content Enrichment.To mitigate the data scarcity and heterogeneity of rare diseases, we plan to augment
both the retrieval corpus and the Question Pool, thereby extending our framework’s generalizability and value in this
challenging domain.
D Method and Evaluation Details
This section details the data cleaning methodologies employed to complement the construction of the retrieval corpus. It
further elucidates the logic governing the curation of the benchmark dataset and offers comprehensive supplementary
information. Lastly, we provide the disaggregated performance metrics for HieraMedTransfer, noting that the aggregated
means of these values constitute the results presented in the main body of the paper.
D.1 Data Cleansing Approaches
Data Acquisition.We adopted the data acquisition methodology outlined in BIOMEDICA [ 52] to collect raw image-
caption pairs. Since BIOMEDICA has already implemented fine-grained unsupervised clustering based on DINOv2 [ 55],
we were able to exclude clinically irrelevant content—such as charts and natural images—by simply filtering based on
the off-the-shelf pseudo-labels.
Multi-Subgraph Decoupling.Scientific literature frequently employs multi-panel figures to serve its illustrative
purposes. However, given that clinical diagnostic images are predominantly presented in a single-panel format, the
abundance of multi-panel content in reference datasets is detrimental to Continual Learning.
a. Data Capturing Model Trains b. Structural Decoupling
Layout  Bag
1 2
3
    Capturing Model
Random
Random
    Capturing Model
A) Immunofluorescence staining of viral hexon 
protein with anti-hexon FITC-conjugated .... 
B) Corresponding Prussian blue stained tissue 
sections demonstrated few ....
Figure 6Overview of the Multi-Subgraph Capture Model: Training
Methodology and Application Scenarios.Drawing inspiration from the work [ 4] of Baghbanzadeh
et al., we recognize that utilizing object detection mod-
els for multi-subgraph partitioning presents a promising
approach. Given that subfigures in scientific literature
typically consist of content with similar domains or
semantics, we utilized single-panel images from our
previously collected dataset for synthesis. Specifically,
we combined images sharing the same coarse-grained
pseudo-labels to generate synthetic multi-panel figures,
resulting in a batched training dataset with object de-
tection annotations. As depicted in Fig.6, we present a
Multi-Subgraph Capture Model tailored for the medical
domain, leveraging the DAB-DETR [51] architecture.
Ultimately, regular expressions were utilized to identify
subfigure indicators (e.g., (1), (a), A). This facilitated the segmentation and realignment of captions based on spatial
layout, given that scientific literature typically follows a fixed left-to-right reading sequence.
17

Table 5Visualization of the nine datasets utilized in HieraMedTransfer. We implemented a design for multi-scale transfer across
both in-domain and out-of-domain settings.
Dataset Example Dataset Name Domain Region/Type Number Classes
RANZCR [64] X-ray Blood Vessel 33665 11
CheXchoNet [9] X-ray Chest 71589 4
PD [3] X-ray Lung 4575 3
Breakhis [65] Patho. Breast 7909 2
Chaoyang [92] Patho. Colonic 6160 4
Nucls [34] Patho. Gastric 33284 2
Eyepacs [30] Fundus Diabetes 35126 5
AIROGS [19] Fundus Glaucoma 101442 2
FARFUM-RoP [1] Fundus ROP 1533 3
D.2 Benchamrk Database Construction
We constructed the MGTIL benchmark based on three key principles:
•There are substantial discrepancies across domains, which vary significantly in terms of imaging modalities and spatial
resolutions.
•The dataset preserves fine-grained intra-domain variations, such as diverse spatial regions and varying object scales,
thereby avoiding severe homogeneity.
•These challenging datasets typify the common hurdles in medical classification tasks, being characterized by a large
number of categories, high task complexity, and the prevalence of few-shot scenarios.
HieraMedTransfer Construction.As detailed in Tab. 5, our experimental evaluation encompasses three distinct
modalities: X-ray, pathological, and fundus images. The intra-domain datasets exhibit multi-level discrepancies,
including variations in anatomical regions, resolutions, and label granularity, which effectively mirrors the complex data
heterogeneity inherent to real-world medical scenario.
MedXtreme Construction.MedXtreme encompasses six medical datasets across distinct domains, characterized by
large label spaces, high task complexity, and significant domain shifts. It is designed to evaluate a model’s capacity for
learning and memory retention on challenging tasks within a continual fine-tuning setting. Notably, the inclusion of a
substantial number of few-shot classes effectively simulates the dilemma of diagnosing rare diseases in clinical practice.
Further details on the dataset are provided in Tab.6.
Detailed Performance Analysis.Tab.7 and Tab.8 detail the results on HieraMedTransfer for Order I and II. Task-specific
metrics follow the ZSCL [ 90] protocol, with averages listed in the ”Average” column as in the main text. Notably, we
supplement these tables with ”Fine-tune” results—representing the theoretical upper bound achieved by fine-tuning
solely on the target dataset. As shown, our method yields the highest average performance among state-of-the-art
competitors and, most notably, performs comparably to the Fine-tune upper bound.
We visualize the accuracy evolution of selected tasks from two distinct orders in Fig.7 (a) and (b). In the context of
Vision-Language Models (VLMs), an optimal Continual Learning strategy is characterized by a“mirrored Z-shaped”
curve. This signifies the effective maintenance of zero-shot capabilities prior to task acquisition and the mitigation of
forgetting subsequent to learning. As demonstrated, our method aligns closely with this ideal profile.
18

Table 6Visualization of the six datasets utilized in MedXtreme. This collection was curated to maximize classification difficulty
while satisfying the data requirements for fine-tuning.
Dataset Example Dataset Name Domain Region/Type Number Classes
AOD [60] Fundus Eye 10000 8
BMC [53] Cell Bone Marrow 171375 21
ISIC2024 [42] Skin Skin 81722 33
NCT100K [38] Patho. Colorectal 100000 9
NIH-Chest-Xray [72] X-ray Chest 112120 15
PITVIS [66] Endo. Pituitary 120024 15
        TASK2:Breakhis                            TASK4:CheXchoNet                       TASK6:FARFUM-RoP                             TASK8:PD
1020304050
12Accuracy
Task Sequence34567891020304050
12Accuracy
Task Sequence345678960708090
50
12Accuracy
Task Sequence345678950606580
55
12Accuracy
Task Sequence3456789607080
75
70
60708090
12Accuracy
Task Sequence34567891020304050
12Accuracy
Task Sequence345678960657075
55
12Accuracy
Task Sequence3456789506090
40
12Accuracy
Task Sequence3456789607080
80
7080                     TASK1:RANZCR                               TASK2:CheXchoNet                           TASK4:Breakhis                             TASK5:Chaoyang(a)  Accuracy Changes of  HieraMedTransfer benchmark in Order I.
(b)  Accuracy Changes of  HieraMedTransfer benchmark in Order II.
Continue-FT WISE-FT ZSCL SND GIFT PRIMED
Figure 7Illustration of classification accuracy changes as tasks are learned on the HieraMedTransfer benchmark in two orders. Our
method consistently exhibits a mirrored Z-shaped pattern.
19

Table 7Detailed Transfer, Avg., and Last scores (%) of different continue training methods on HieraMedTransfer benchmark in
Order I. Red Background&boldindicate best results.
Method
RANZCR [64]
CheXchoNet [9]
PD [3]
Breakhis [65]
Chaoyang [92]
NuCLS [34]
Eyepacs [30]
AIROGS [19]
FARFUM-RoP [1] Average
Zero-shot 16.44 28.34 52.07 53.88 52.65 57.45 62.28 96.46 56.49 52.90
Fine-tune 53.13 86.55 95.64 97.35 80.91 97.03 78.33 97.55 79.87 85.15
Transfer
Continual FT 7.85 39.22 47.33 54.85 56.50 57.65 91.52 56.49 51.43
𝑙2baseline 13.09 38.24 51.41 54.49 55.68 64.34 95.40 56.48 53.64
LwF [48] 31.89 43.6852.5154.13 56.49 19.96 34.30 56.49 43.68
iCaRL [61] 7.70 32.89 54.49 55.5760.6058.26 96.67 56.41 52.82
WiSE-FT [75] 13.33 40.85 49.31 54.37 55.85 60.66 94.51 56.49 53.17
ZSCL [90] 34.88 47.17 51.08 54.41 57.2664.4395.68 56.49 57.68
MoE-CL [83] 32.61 51.36 50.14 53.96 56.40 61.72 92.87 55.38 56.81
SND [85] 7.85 40.20 46.95 54.77 58.21 58.65 92.98 56.41 52.00
DIKI [69] 21.87 47.51 51.03 54.96 58.36 63.82 96.65 56.48 56.34
GIFT [76] 15.42 42.57 52.21 55.46 52.27 59.25 93.77 56.49 53.43
PRIMED𝑢𝑛𝑖 27.4652.4050.91 54.98 55.77 62.73 96.21 56.49 57.12
PRIMED𝑑𝑦𝑛 39.09 48.38 51.24 55.58 55.42 63.68 96.72 56.49 58.33
Avg.
Continual FT 31.69 70.13 80.90 75.81 66.16 72.41 59.83 91.84 59.38 67.57
𝑙2baseline 27.71 76.31 77.25 76.05 66.25 71.94 66.82 95.35 58.94 68.51
LwF [48] 8.55 67.76 68.70 68.84 60.52 70.07 37.36 47.07 59.45 54.26
iCaRL [61] 49.69 77.56 71.53 76.41 64.9276.0064.4296.7859.09 70.71
WiSE-FT [75] 35.33 74.52 81.38 76.46 66.83 71.93 63.60 94.79 59.45 69.37
ZSCL [90] 43.91 78.98 80.39 72.74 64.74 70.76 67.67 95.90 59.09 70.46
MoE-CL [83] 33.07 80.5884.9277.92 65.58 70.87 67.40 96.84 59.43 70.73
SND [85] 25.77 71.23 82.21 75.96 65.88 73.43 60.05 93.50 59.02 67.45
DIKI [69] 41.84 79.50 80.74 75.70 64.81 72.58 67.61 96.28 59.41 70.94
GIFT [76] 42.97 72.68 80.37 75.73 66.92 71.39 63.57 95.95 58.29 69.76
PRIMED𝑢𝑛𝑖 50.49 79.34 83.95 78.41 66.52 72.98 66.63 96.2559.4572.67
PRIMED𝑑𝑦𝑛 50.95 80.83 83.35 79.82 67.71 72.71 67.73 95.61 59.30 73.11
Last
Continual FT 14.49 75.50 90.41 87.10 72.49 85.46 40.84 88.32 82.47 70.79
𝑙2baseline 15.88 81.04 89.11 87.48 72.98 85.40 63.86 93.57 77.92 74.14
LwF [48] 1.92 75.67 70.59 72.06 64.40 67.56 64.31 86.23 83.12 65.10
iCaRL [61] 48.6686.2379.21 73.70 65.0594.62 75.9396.12 80.52 77.78
WiSE-FT [75] 24.51 79.94 90.63 88.87 73.79 86.42 55.48 94.28 83.12 75.23
ZSCL [90] 41.75 81.94 87.58 80.91 71.84 84.41 74.08 96.08 79.87 77.61
MoE-CL [83] 38.80 79.57 85.71 82.83 71.94 87.27 67.61 94.27 77.25 76.14
SND [85] 11.95 74.26 91.94 87.10 71.36 86.51 35.29 92.99 79.87 70.14
DIKI [69] 41.29 84.16 91.60 86.11 72.10 84.97 70.64 91.56 74.47 77.43
GIFT [76] 37.18 65.71 84.75 86.22 75.89 92.67 64.82 96.34 72.73 75.15
PRIMED𝑢𝑛𝑖 49.7884.77 91.94 91.66 74.60 93.30 70.64 95.9683.1281.75
PRIMED𝑑𝑦𝑛 47.07 84.80 92.37 92.92 76.21 92.31 74.74 96.40 81.82 82.07
20

Table 8Detailed Transfer, Avg., and Last scores (%) of different continue training methods on HieraMedTransfer benchmark in
Order II. Red Background&boldindicate best results.
Method
AIROGS [19]
Breakhis [65]
Chaoyang [92]
CheXchoNet [9]
Eyepacs [30]
FARFUM-RoP [1]
NuCLS [34]
PD [3]
RANZCR [64] Average
Zero-shot 96.46 53.88 52.65 28.34 62.28 56.49 57.45 52.07 16.44 52.90
Fine-tune 97.55 97.35 80.91 86.55 78.33 79.87 97.03 95.64 53.13 85.15
Transfer
Continual FT 56.01 56.15 28.83 64.18 56.49 55.10 52.63 11.98 47.67
𝑙2baseline 55.12 55.82 18.58 64.80 56.62 54.15 53.52 7.40 45.75
LwF [48] 57.65 55.82 19.78 61.65 55.19 57.56 52.94 14.42 46.88
iCaRL [61] 53.73 54.69 38.3968.5156.4957.9248.80 6.57 48.14
WiSE-FT [75] 55.25 55.74 28.09 67.45 56.49 54.44 54.74 14.11 48.29
ZSCL [90] 55.25 55.10 9.42 67.86 51.43 54.31 56.83 11.36 45.20
MoE-CL [83] 52.87 55.50 28.78 63.12 55.93 53.29 53.09 18.25 47.60
SND [85]56.0156.40 24.34 64.42 55.84 56.04 46.31 15.96 46.92
DIKI [69] 52.06 55.02 20.54 64.76 55.63 54.77 49.19 20.57 46.57
GIFT [76] 54.36 54.86 18.51 67.72 56.49 55.06 51.36 15.95 46.79
PRIMED𝑢𝑛𝑖 53.98 54.86 18.90 66.23 56.49 53.1856.92 22.6247.90
PRIMED𝑑𝑦𝑛 53.78 56.40 34.66 64.28 56.89 52.09 53.62 16.44 48.52
Avg.
Continual FT 79.46 87.18 71.84 61.17 48.53 67.59 68.67 55.94 16.55 61.88
𝑙2baseline 94.25 84.67 69.83 62.98 62.75 64.79 67.26 56.77 12.18 63.94
LwF [48] 89.64 72.64 63.22 53.50 63.82 65.87 70.31 59.04 17.45 61.72
iCaRL [61] 96.93 87.81 68.1670.34 73.5861.2570.6954.01 11.20 66.00
WiSE-FT [75] 92.30 86.91 72.14 64.70 57.51 66.66 66.65 57.98 18.37 64.80
ZSCL [90] 96.70 82.92 69.04 60.61 71.55 56.85 66.8664.6615.92 65.01
MoE-CL [83] 95.88 88.51 69.98 63.57 68.41 64.76 65.11 59.62 18.73 66.06
SND [85] 81.36 86.31 71.20 58.23 48.53 63.49 69.46 50.37 20.05 61.00
DIKI [69] 96.95 86.80 69.94 61.02 69.17 62.86 65.44 62.69 16.39 65.70
GIFT [76] 95.44 84.34 72.19 62.61 61.64 65.22 68.37 55.46 19.85 65.01
PRIMED𝑢𝑛𝑖 96.51 88.12 72.08 62.21 69.54 66.74 67.33 60.78 25.81 67.68
PRIMED𝑑𝑦𝑛 97.00 89.06 72.19 69.06 69.34 67.68 66.46 61.24 20.32 68.04
Last
Continual FT 39.17 86.98 71.20 43.22 17.29 81.17 95.46 40.0953.0958.63
𝑙2baseline 78.62 86.47 66.02 81.80 38.91 66.23 91.59 41.18 50.45 66.81
LwF [48] 59.47 67.89 63.92 40.52 48.05 74.68 95.46 59.69 45.65 61.70
iCaRL [61] 96.79 84.07 64.72 86.2377.3362.99 94.82 49.67 48.26 73.88
WiSE-FT [75] 76.65 87.10 74.76 67.10 23.61 81.17 86.00 44.23 52.46 65.90
ZSCL [90] 96.67 84.96 70.71 86.13 74.51 45.45 91.17 89.98 52.33 76.88
MoE-CL [83] 81.34 85.95 70.27 76.56 64.75 71.58 91.47 75.99 50.02 74.21
SND [85] 47.02 87.99 72.33 26.00 9.71 55.19 95.55 33.33 52.76 53.32
DIKI [69] 90.07 85.97 73.11 78.35 68.66 75.32 90.84 80.31 48.03 76.74
GIFT [76] 90.59 84.83 75.08 78.87 41.24 78.57 94.83 46.84 51.04 71.32
PRIMED𝑢𝑛𝑖 96.9088.37 75.24 78.33 70.66 78.5796.4057.12 51.31 76.99
PRIMED𝑑𝑦𝑛 91.93 89.51 75.73 86.31 68.61 81.17 95.01 91.29 51.31 81.21
21

Table 9Ablation study of different modules on HieraMedTransfer and MedXtreme. Red Backgroundindicates the full model.
+CKT +CMC +DFGHieraMedTransfer I HieraMedTransfer II MedXtreme I MedXtreme II
Transfer Avg. Last Transfer Avg. Last ACC AUC BWT ACC AUC BWT
✓ 56.2 71.882.6 46.7 67.8 81.0 66.7 87.1 -4.2 65.9 86.3 -5.3
✓ 54.3 68.9 78.9 49.5 67.2 76.4 64.9 85.4 -7.6 60.4 83.7 -13.2
✓ ✓ 56.9 71.6 82.2 48.2 67.9 81.0 66.5 85.9 -4.8 64.9 85.7 -6.8
✓ ✓ 57.2 72.8 82.5 46.8 67.8 80.9 66.7 87.3 -4.3 65.8 86.1 -5.4
✓ ✓ 56.7 70.2 77.1 50.067.4 77.2 65.2 85.2 -7.1 60.4 82.8 - 13.3
✓ ✓ ✓ 58.3 73.1 82.1 48.5 68.0 81.2 68.6 87.4 -2.7 68.1 86.3 -3.4
E Ablation Study
To validate the effectiveness of our experimental settings at all levels, we performed extensive ablation studies involving
four sequences on our two proposed benchmarks. The analysis is organized as follows: experimental setup, module-level
ablation, hyperparameter and component-level ablation, and Dynamic Retrieval analysis.
E.1 Full Settings
To ensure reproducibility, we detail the key configurations and experimental settings for training our model as follows:
•Batch Size and Label Smoothing:We employ a batch size of 64 per GPU and apply label smoothing of 0.2. Notably,
fine-tuning is fixed at 1,000 iterations across all datasets; for datasets with insufficient samples, the training data is
cycled to meet this requirement.
•Learning Rate:A unified learning rate of 1×10−5is applied across all regularization, replay, and distillation
methods [ 48,61,75,76,85,90]. For approaches based on LoRA [ 83] or Prompt Tuning [ 69], we strictly adhere to the
hyperparameter settings outlined in their papers.
•Detailed Configurations:In the following sections, we present a comprehensive ablation study covering all relevant
components and hyperparameters. For clarity, the default settings adopted in our method are highlighted with a
Red Background.
E.2 Modular Level Analysis
As shown in Tab.9, we conducted module-level ablation studies across all benchmarks and sequences. Encouragingly,
the results remain fully consistent with the conclusions presented in the main text. This demonstrates the exceptional
robustness of our method and the synergistic coupling between modules.
E.3 Component and Hyperparameter Analysis
Tab.10 presents the ablation study demonstrating robustness at both the component and parameter levels, while
consistently maintaining superior performance. Notably, under the challenging task scenarios simulated by MedXtreme,
our retrieval method achieved a significantly larger performance margin compared to other approaches. This trend
aligns with the behaviors observed in dynamic recall on uniformly distributed reference datasets. This suggests that
in challenging scenarios or complex clinical settings, retrieval mechanisms with higher quality and finer granularity
possess significant efficacy and potential.
E.4 Dynamic Retrieval Dataset Analysis
The Dynamic Retrieval component is the cornerstone of PRIMED and constitutes the fundamental difference between
our method and existing approaches. Two specific aspects warrant further investigation. First, akin to other methods
relying on reference datasets, the size of the dataset presents a critical trade-off. Insufficient capacity risks compromising
generalization and diversity, while excessive size imposes a prohibitive computational and storage overhead. Since prior
studies have demonstrated that performance tends to plateau beyond a certain threshold, our objective is to identify this
optimal saturation point illustrated in Fig.8. Next, building on the determined peak number, we investigated the ratios for
dynamic retrieval. Operating under the premise that task weights should exceed domain weights, which in turn should
22

Table 10Ablation study on different components and hyperparameters. Red Backgroundindicates optimal settings.
Comp./Hparam. HieraMedTransfer I HieraMedTransfer II MedXtreme I MedXtreme II
Aspect Detail Transfer Avg. Last Transfer Avg. Last ACC AUC BWT ACC AUC BWT
Initial CLIP 57.9 70.7 78.2 47.7 66.9 80.1 60.5 80.7 -9.5 60.3 78.9 -11.2
Teacher Last CLIP 58.3 73.1 82.1 48.5 68.0 81.2 68.6 87.4 -2.7 68.1 86.3 -3.4
WISE(0.5) 58.1 71.4 79.1 48.3 67.1 80.5 63.5 83.1 -5.6 62.7 82.0 -8.1
Image-only 57.7 72.4 81.3 47.9 67.6 80.8 67.6 87.0 -3.9 67.0 86.2 -4.5
KD Loss Text-only 58.2 73.0 80.5 49.867.9 80.4 65.5 85.8 -6.3 63.2 85.3 -9.2
Contra. 58.3 73.1 82.1 48.5 68.0 81.2 68.6 87.4 -2.7 68.1 86.3 -3.4
𝛼=0.5 58.3 72.8 81.3 48.767.9 80.6 66.5 86.9 -5.1 66.7 85.4 -4.4
CKT Scale 𝛼=1 58.3 73.1 82.1 48.5 68.0 81.2 68.6 87.4 -2.7 68.1 86.3 -3.4
𝛼=1.5 58.3 73.0 81.9 48.0 67.8 81.0 66.2 87.4 -4.9 66.3 85.6 -5.1
𝛽=0.0 57.2 72.882.5 46.8 67.8 80.9 66.7 87.3 -4.3 65.8 86.1 -5.4
CMC Scale 𝛽=0.25 58.3 73.1 82.1 48.5 68.0 81.2 68.6 87.4 -2.7 68.1 86.3 -3.4
𝛽=0.5 58.1 72.5 81.4 48.2 67.4 77.3 66.8 85.7 -4.6 64.2 84.8 -7.9
𝑙2 57.2 72.9 81.8 48.2 67.9 81.0 66.5 86.9 -4.8 64.9 86.4 -6.9
Reg. EWC 57.2 71.3 82.1 48.0 66.881.4 67.3 87.2 -3.7 67.286.9-3.5
DFG 58.3 73.1 82.1 48.5 68.0 81.2 68.6 87.4 -2.7 68.1 86.3 -3.4
BM25 55.1 72.2 81.8 47.5 67.5 79.4 67.4 86.8 -3.4 64.8 85.5 -6.5
RAG Embedding 56.1 72.5 81.5 47.3 67.2 76.7 66.3 86.9 -4.8 63.9 85.4 -7.5
Hierarchical 58.3 73.1 82.1 48.5 68.0 81.2 68.6 87.4 -2.7 68.1 86.3 -3.4
exceed general weights, we employed a grid search to identify the optimal ratios. The quantitative results regarding the
dataset capacity saturation are detailed in Tab. 11, while the outcomes of the grid search for optimal retrieval ratios are
tabulated in Tab. 12.
Accuracy
HieraMedTransfer Order I
HieraMedTransfer Order II
NumberMedXtreme Order I
MedXtreme Order II
Figure 8Peak Performance of Dynamic Retrieval across DatasetsThe experimental results high-
light distinct requirements for
recall versus generalization
across different tasks. Specif-
ically, for intra- and cross-
domain transfer tasks such as
HieraMedTransfer, enhanced
generalization is pivotal for han-
dling diverse scenarios. Con-
versely, in high-difficulty benchmarks like MedXtreme, models require an extensive, potentially iterative review of
representative exemplars. This is an intriguing finding, as it parallels human cognitive processes that combine long-term
retention with short-term intensive reinforcement. Indeed, many characteristics of model memory appear to mirror those
inherent to human memory.
F Backbone Selection and Generalizability
Prior to selecting the specific backbones, we briefly review the list of candidate models, providing a concise overview of
these contrastive learning-based foundation models.
•BiomedCLIP [ 88]is a multimodal foundation model pretrained on PMC-15M, a large-scale dataset of 15 million
image-text pairs sourced from 4 million scientific articles
•MMKD-CLIP [ 71]is a generalist biomedical foundation model developed via multi-teacher knowledge distillation,
utilizing 19.2 million image-text feature pairs synthesized by 9 expert models from the PMC-OA dataset.
•BMC-CLIP [ 52]is trained on the large-scale BIOMEDICA dataset, which includes over 24M image-text pairs from
over 6M open-access scientific articles.
•UniMed-CLIP [ 39]:is a unified vision-language model trained on UniMed, a large-scale open-source dataset of 5.3
23

Table 11Dynamic Retrieval Analysis in HieraMedTransfer.
Ratio HieraMedTransfer I HieraMedTransfer II
𝑏:𝑐 Trans. Avg. Last Trans. Avg. Last
10:90 58.0 72.8 81.7 48.3 67.8 80.8
20:80 58.3 73.0 82.0 48.2 67.9 81.1
30:80 58.3 73.1 82.1 48.5 68.0 81.2
40:80 58.3 73.0 82.0 48.4 67.8 79.8
50:50 58.3 73.0 81.7 48.5 68.0 80.2
60:40 58.2 72.9 82.1 48.0 67.9 80.8
80:20 58.2 73.0 81.4 48.4 68.0 81.2
90:10 58.2 73.1 81.5 48.1 67.6 78.6Table 12Dynamic Retrieval Analysis in MedXtreme.
Ratio MedXtreme I MedXtreme II
𝑏:𝑐 ACC AUC BWT ACC AUC BWT
100:100 68.4 87.0-2.4 67.9 85.9 -3.5
0:100 68.2 86.5 -3.2 67.7 85.7 -3.9
70:70 68.6 86.9 -2.7 68.1 85.7 -3.4
70:90 68.4 87.0 -2.9 67.8 85.8 -3.5
70:100 68.6 87.4 -2.7 68.1 86.3 -3.4
50:100 68.5 87.3 -2.8 68.1 86.3 -3.4
50:75 68.3 87.1 -3.0 67.7 86.0 -4.2
90:100 68.2 86.9 -2.5 67.9 85.7 -3.7
Table 13Experimental Results of Continual Learning on Backbones Other than BiomedCLIP
Architecture HieraMedTransfer I HieraMedTransfer II MedXtreme I MedXtreme II
Backbone Method Transfer Avg. Last Transfer Avg. Last ACC AUC BWT ACC AUC BWT
Continual FT 45.8 66.9 79.4 45.7 66.8 79.5 65.7 87.5 -7.1 64.5 85.7 -8.6
WiSE-FT [75] 46.4 67.1 80.2 46.4 67.1 80.0 64.8 87.3 -4.0 62.4 86.0 -6.9
MMKD [71] ZSCL [90] 58.669.3 77.8 44.8 65.0 79.8 57.4 81.4 -8.6 56.0 80.9 -10.4
GIFT [76] 49.6 68.3 80.5 49.6 68.3 80.4 69.7 88.2 -1.8 68.4 88.1 -3.1
PRIMED𝑑𝑦𝑛 50.7 69.4 80.7 50.7 69.4 80.6 70.4 88.3 -1.5 70.5 88.1 -1.3
Continual FT 45.0 66.4 79.8 44.9 66.4 80.0 61.0 83.4 -10.5 58.8 82.6 -13.7
WiSE-FT [75] 44.0 66.5 81.5 44.0 66.5 81.4 57.9 83.2 -9.1 61.6 86.5 -5.4
UniMed [39] ZSCL [90] 44.9 66.5 81.4 42.5 63.8 81.9 63.2 85.0 -6.3 59.9 83.6 -10.2
GIFT [76] 44.2 66.7 82.5 44.2 66.7 82.4 67.188.7-3.6 67.0 87.6 -3.5
PRIMED𝑑𝑦𝑛 45.0 67.1 83.1 45.1 67.1 83.2 67.4 88.4 -3.3 67.2 88.3 -3.3
Continual FT 34.9 59.1 71.1 34.2 58.6 54.4 61.5 83.7 -17.2 47.3 82.5 -34.1
WiSE-FT [75] 34.8 60.3 70.7 34.9 61.6 72.6 64.2 85.1 -12.4 57.3 84.9 -20.8
CLIP [59] ZSCL [90] 35.7 61.1 78.0 35.5 61.0 78.5 63.0 82.3 -11.0 56.2 82.9 -19.2
GIFT [76] 36.261.7 73.9 36.262.6 75.5 71.0 87.5 -6.0 70.9 88.7 -5.9
PRIMED𝑑𝑦𝑛 35.7 62.6 84.6 35.6 62.6 84.0 73.8 89.1 -2.6 73.7 89.4 -2.8
million image-text pairs spanning six imaging modalities (X-ray, CT, MRI, Ultrasound, Pathology, Fundus).
•CLIP [ 59]:is a multimodal foundation model pretrained on WIT-400M, a large-scale dataset of 400 million image-text
pairs collected from a variety of publicly available sources on the internet.
F.1 Justification for the BiomedCLIP Backbone
Our choice of BiomedCLIP as the primary backbone is motivated by several factors, outlined below in descending order
of significance. Crucially, we must underscore that this selection was not predicated solely on performance metrics.
Indeed, considerations such as the guarantee against data contamination and the maturity of the architectural framework
took precedence over raw performance.
PD
CheXchoNetRANZCR
FARFUM-RoP
EyepacsNuCLSChaoyangBreakhis
ISIC2024BMCAOD
PITVISNIH-Chest-XrayNCT100K
AIROGS
BiomedCLIP MMKD-CLIP BMC-CLIP UniMed-CLIP CLIPHieraMedTransfer                                                         MedXtreme
Figure 9The zero-shot capabilities of 5 backbones across Hier-
aMedTransfer and MedXtreme are depicted in radar chart format.Data Security.Admittedly, while all the afore-
mentioned methods utilize open-source datasets, only
BiomedCLIP and BMC-CLIP feature a comprehensive
data acquisition architecture. This distinction fundamen-
tally ensures data integrity and reproducibility while
preventing data contamination. Although MMKD-CLIP
exhibits impressive experimental performance, it is de-
rived from multi-model distillation, making it difficult to
fully verify its data provenance. Therefore, establishing
a more controllable baseline is of paramount importance
to our work.
24

Zero-shot performance.Fig.9 presents the zero-shot results of various models on the HieraMedTransfer and
MedXtreme benchmarks. For HieraMedTransfer, it is essential that the backbone exhibits a reasonable baseline of
zero-shot capability; otherwise, the subsequent transferability evaluation would lack validity. BiomedCLIP demonstrates
consistent performance across all datasets without exhibiting anomalous outlier peaks, establishing it as a highly robust
and reliable candidate. In contrast, all models yield suboptimal performance on MedXtreme. Consequently, absolute
performance metrics are of secondary importance compared to the potential risk of data contamination. In this regard,
BiomedCLIP serves as a good choice.
Model Architecture.We favored a mature architecture that fits our specific demands; specifically, BiomedCLIP
features a well-developed fine-tuning and post-training ecosystem. Additionally, we aimed to maximize experimental
comparability by aligning with the ViT-B configuration used in natural image studies (e.g., ZSCL). Therefore, absent any
distinct performance benefits, the ViT-L versions of BMC-CLIP and UniMed-CLIP were not selected as backbones for
the main experiments.
F.2 Generalizable SOTA Performance
Although we consider BiomedCLIP to be the most intuitively suitable backbone, we also evaluated other ViT-B based
backbones, as shown in Tab.13. Our method consistently achieved superior performance, demonstrating the effectiveness
of our proposed strategy.
G Retrieval Visualization
Echoing the main text, we underscore the distinct superiority of our retrieval approach in terms of intuitive visualization.
Primarily, the intrinsic mechanism of multimodal retrieval endows our method with strong semantic disentanglement
capabilities. A prime example is in dentistry, where our model clearly discriminates between OCT scans and natural
images, despite their high textual semantic overlap. Furthermore, our approach exhibits enhanced recall precision, moving
beyond the rigid constraints of keyword matching. Since our data source relies heavily on multi-subgraph disentanglement,
as detailed above, we effectively filter out cases of failed disentanglement or conceptual ambiguity—providing a robust
guarantee of effectiveness. Ultimately, we are encouraged to observe that the retrieved content demonstrates both
generalizability and hierarchical progression. By moving beyond isolated disease categories to account for the
holistic connections between diseases, lesions, and subtypes, we believe this property is pivotal in improving model
memorization.
25

A thermal imaging photo of 
inflammatory arthritis 
joint temperature mappingCaption     Image    Ours:TOP1
Caption     
Pre-treatment ... of ... 
temperature than did 
the non-affected side ... 
the stifle joint. Image    
Caption     Image    Caption     
Infrared thermal 
imaging of the knee of 
a patient with knee 
osteoarthritis.
Caption     
...of Purtscher-Like 
Retinopathy in a 29-
Year-Old Female. A 
photo of her skin. Image    Image    
A fundus photo of 
Wet Age-related 
Macular Degeneration Caption     
Fundus photograph of 
the ... age-related 
macular degeneration...Image    Ours:TOP1
Caption     
Example of age-related 
macular degeneration 
seen on fundus camera Image    
Caption     
Color fundus photo 
showing clinical 
features of polypoidal 
choroidal vasculopathyImage    Caption     
Fundus photograph ... of 
wet-type age-related 
macular degeneration.
Caption     
.Age-related Macular 
Degeneration (AMD): 
Due to hemorrhage 
and .....Image    Image    Ours:TOP2
W/O Rerank and GateOurs:TOP3
BM25
Ours:TOP2 Ours:TOP3
W/O Rerank and Gate BM25
Post-treatment ... in the 
body temperatures 
between the affected 
and non-affected sides ...Pre-treatment imaging 
of the affected side 
showed a higher body 
temperature than
A CT photo of Kidney TumorCaption     Image    Ours:TOP1
Caption     
Abdominal CT showing a 
tumour in the left 
kidneyImage    
Caption     Image    Caption     
CT image of a right 
kidney tumor.
Caption     
Kidney microscopic 
image of HN rats. HN, 
hyperuricaemic 
nephropathy.Image    Image    
An intraoral radiograph photo 
of dental implantCaption     
Intraoral radiography: 
defect after implant 
removalImage    Ours:TOP1
Caption     
Intra-oral Rx showing 
implant with abutmentsImage    
Caption     
Intraoral radiography: 
implant placementImage    Caption     
Caption     
An intraoral image ... 
postoperatively 
showing ... of the 
keratinized gingiva.Image    Image    Ours:TOP2
W/O Rerank and GateOurs:TOP3
BM25
Ours:TOP2 Ours:TOP3
W/O Rerank and Gate BM25
CT scan showing large 
tumor of the right 
kidney.transversal view of a 
9–cm tumor in the 
upper pole of the left 
kidney.
Intraoral radiography: 
implant placementFigure 10Qualitative comparison with state-of-the-art methods. Our method achieves superior performance in visual correction
and textual precision. By explicitly aligning the hierarchical content within questions with the retrieved data, our method achieves
optimal fine-grained retrieval performance. Furthermore, leveraging visual-level retrieval capabilities allows our approach to prioritize
complete and high-quality images rather than relying solely on textual cues. This capability enhances the dynamic retrieval database’s
distillation.
26