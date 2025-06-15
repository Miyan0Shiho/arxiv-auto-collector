# MrM: Black-Box Membership Inference Attacks against Multimodal RAG Systems

**Authors**: Peiru Yang, Jinhua Yin, Haoran Zheng, Xueying Bai, Huili Wang, Yufei Sun, Xintian Li, Shangguang Wang, Yongfeng Huang, Tao Qi

**Published**: 2025-06-09 03:48:50

**PDF URL**: [http://arxiv.org/pdf/2506.07399v1](http://arxiv.org/pdf/2506.07399v1)

## Abstract
Multimodal retrieval-augmented generation (RAG) systems enhance large
vision-language models by integrating cross-modal knowledge, enabling their
increasing adoption across real-world multimodal tasks. These knowledge
databases may contain sensitive information that requires privacy protection.
However, multimodal RAG systems inherently grant external users indirect access
to such data, making them potentially vulnerable to privacy attacks,
particularly membership inference attacks (MIAs). % Existing MIA methods
targeting RAG systems predominantly focus on the textual modality, while the
visual modality remains relatively underexplored. To bridge this gap, we
propose MrM, the first black-box MIA framework targeted at multimodal RAG
systems. It utilizes a multi-object data perturbation framework constrained by
counterfactual attacks, which can concurrently induce the RAG systems to
retrieve the target data and generate information that leaks the membership
information. Our method first employs an object-aware data perturbation method
to constrain the perturbation to key semantics and ensure successful retrieval.
Building on this, we design a counterfact-informed mask selection strategy to
prioritize the most informative masked regions, aiming to eliminate the
interference of model self-knowledge and amplify attack efficacy. Finally, we
perform statistical membership inference by modeling query trials to extract
features that reflect the reconstruction of masked semantics from response
patterns. Experiments on two visual datasets and eight mainstream commercial
visual-language models (e.g., GPT-4o, Gemini-2) demonstrate that MrM achieves
consistently strong performance across both sample-level and set-level
evaluations, and remains robust under adaptive defenses.

## Full Text


<!-- PDF content starts -->

arXiv:2506.07399v1  [cs.CV]  9 Jun 2025MrM: Black-Box Membership Inference Attacks
against Multimodal RAG Systems
Peiru Yang∗Jinhua Yin∗Haoran Zheng†Xueying Bai†
Huili Wang∗Yufei Sun†Xintian Li∗Shangguang Wang†
Yongfeng Huang∗Tao Qi‡
Abstract
Multimodal retrieval-augmented generation (RAG) systems enhance large vision-
language models by integrating cross-modal knowledge, enabling their increasing
adoption across real-world multimodal tasks. These knowledge databases may
contain sensitive information that requires privacy protection. However, multi-
modal RAG systems inherently grant external users indirect access to such data,
making them potentially vulnerable to privacy attacks, particularly membership
inference attacks (MIAs). Existing MIA methods targeting RAG systems predomi-
nantly focus on the textual modality, while the visual modality remains relatively
underexplored. To bridge this gap, we propose MrM, the first black-box MIA
framework targeted at multimodal RAG systems. It utilizes a multi-object data
perturbation framework constrained by counterfactual attacks, which can concur-
rently induce the RAG systems to retrieve the target data and generate information
that leaks the membership information. Our method first employs an object-aware
data perturbation method to constrain the perturbation to key semantics and ensure
successful retrieval. Building on this, we design a counterfact-informed mask
selection strategy to prioritize the most informative masked regions, aiming to
eliminate the interference of model self-knowledge and amplify attack efficacy.
Finally, we perform statistical membership inference by modeling query trials to
extract features that reflect the reconstruction of masked semantics from response
patterns. Experiments on two visual datasets and eight mainstream commercial
visual-language models (e.g., GPT-4o, Gemini-2) demonstrate that MrM achieves
consistently strong performance across both sample-level and set-level evaluations,
and remains robust under adaptive defenses.
1 Introduction
As a key augmentation strategy for LLMs, retrieval-augmented generation (RAG) has recently been
extended to the visual modality, enabling broader applicability in multimodal AI tasks [ 1–3]. By
incorporating visual modalities, RAG systems can retrieve external knowledge that complements the
visual input and helps reduce hallucinations in large vision-language models (LVLMs) [ 4,5]. Recent
advancements demonstrate the emerging role of multimodal RAG in enabling LVLMs to dynamically
integrate knowledge for real-world applications, such as intelligent medical AI systems [6–8].
For the effectiveness of RAG systems, some private-domain databases are incorporated to support
vertical inference and complex reasoning [ 9]. These knowledge bases often contain private or
proprietary data that are essential for supporting complex downstream tasks, while such data can be
∗Tsinghua University. Email: ypr21@mails.tsinghua.edu.cn
†Beijing University of Posts and Telecommunications.
‡Beijing University of Posts and Telecommunications. Corresponding author. Email: taoqi.qt@gmail.com
Preprint. Under review.

highly sensitive and should be safeguarded with robust privacy protections [ 10,11]. Yet, the RAG
paradigm inherently introduces an indirect exposure risk: the knowledge base provides information
to the generation model, which then produces responses accessible to external users. In doing
so, the RAG system establishes a bridge between internal sensitive data and external adversaries,
enabling interactions that may inadvertently leak private content. This indirect access pathway creates
new vulnerabilities, allowing adversaries to mount privacy attacks against the underlying database,
particularly membership inference attacks (MIAs), which seek to reveal whether specific samples
were part of the original database [12–17].
Existing research on MIAs against RAG has primarily focused on text-only modality, employing
various methodologies to determine whether a target sample exists in the retrieval corpus [ 18–21].
For instance, Li et al. [19] develop an MIA approach that analyzes semantic similarity and perplexity
between target samples and RAG-generated content to infer database membership. In conclusion,
the paradigm of these methods involves providing fragments of target data and then comparing the
similarity between the output and the original data. However, LVLMs process both textual and visual
inputs while typically generating text-only outputs [ 22–24]. This asymmetry introduces a challenge
of modality transfer: inferring the membership status of visual data requires reasoning over purely
textual responses, without direct access to visual features in the output. Hence, these text-centric MIA
methods cannot be directly transferred to LVLMs with multimodal RAG. Besides, a new challenge
lies in the balance between ensuring the successful retrieval of target data and guiding the model
generation towards revealing membership information. The solutions to the two aforementioned core
challenges are crucial for MIAs specifically designed for multimodal RAG systems.
Therefore, we propose MrM, a multi-object data perturbation framework constrained by counterfactual
attacks, which is the first black-box MIA framework targeted at multimodal RAG systems. Its core
idea is perturbing target samples and analyzing whether the textual responses implicitly reconstruct
the disrupted semantics. In this way, our method the semantics of the text and visual modalities
precisely through object detection to tackle the challenge of cross-modal membership inference.
Moreover, masking objects can minimize the affected region, thereby enhancing the effectiveness of
attacks on retrieval, while perturbing the independent and complete semantics to strengthen attacks
on generation. Specifically, an object-aware data perturbation approach is employed to strategically
disrupt visual semantics by masking detected entities using object detection models such as SAM[ 25].
This approach ensures that key features are disrupted while still allowing relevant data to be retrieved
if it exists in the database. It is followed by a counterfact-informed mask selection strategy, where we
quantify the informativeness of each perturbation. We prioritize masks that maximize discriminative
gaps by analyzing probability distributions and confidence differentials of a counterfactual proxy
model. This strategy aims to eliminate the interference of the self-knowledge of LVLMs, thereby
preventing the reconstruction of information for non-database images during the generation phase.
Finally, we perform statistical membership inference by modeling query trials to analyze whether the
textual responses implicitly reconstruct the disrupted semantics.
In conclusion, the contributions of our method are as follows:
•We introduce the first black-box MIA framework for multimodal RAG systems, highlighting
vulnerabilities in the privacy protection of multimodal databases.
•We propose a unified MIA framework that addresses the cross-modal alignment issue and
enables concurrent attacks in both the retrieval and generation phases.
•We validate our framework through comprehensive experiments across two visual datasets
and eight mainstream commercial LVLMs, demonstrating consistent strong performance
and robustness against adaptive defense strategies.
2 Related Work
Multimodal RAG . Since the emergence of cross-modal alignment models like CLIP [ 26], ViL-
BERT [ 27], and BLIP [ 28], multimodal RAG has emerged as a critical solution to address the
inherent limitations of unimodal frameworks in processing cross-modal correlations, where isolated
text-based retrieval fails to capture the intricate interplay between visual semantics and linguistic
context required for complex vision-language reasoning[ 1–3,29]. Chen et al. [1]propose a multi-
modal retrieval-augmented transformer that enhances language generation by accessing an external
2

③Statistical Membership Inference
Accept or
Reject 𝑯𝑯𝑯
Target 
Blackbox 
Multimodal 
RAG System
�𝐻𝐻𝑯:𝐼𝐼∈𝑫𝑫𝑫𝑫
𝐻𝐻𝐻:𝐼𝐼∉𝑫𝑫𝑫𝑫𝑥𝑥~Geometric 𝑝𝑝
if 𝐼𝐼∈𝑫𝑫𝑫𝑫①Object -Aware Data Perturbation
Target Data 𝐼𝐼
 Detection Model
Cat Apple
Apple AppleApple
 Data 
Augmentation
Objects
Cat Apple···
Apple
Multimodal 
Database (𝑫𝑫𝑫𝑫)
CLIP
Multimodal 
RetrieverTarget VLM
LLMText
Image
②Counterfact -Informed Mask Selection
𝑝𝑝𝑐𝑐=𝒫𝒫𝑐𝑐
𝒫𝒫𝑡𝑡𝑡𝑡𝑡𝑡−𝑘𝑘={𝑝𝑝𝑖𝑖}𝑖𝑖=1𝑘𝑘
Top-k Probabilities
Δ𝑝𝑝=max𝒫𝒫−𝑝𝑝𝑐𝑐
ℋ=−∑𝑖𝑖=1𝑉𝑉𝑝𝑝𝑖𝑖𝑙𝑙𝑙𝑙𝑙𝑙𝑝𝑝𝑖𝑖
Feature ExtractionUnsupervised
&
Lightweight
𝑉𝑉𝑝𝑝
Apple
Counterfact Proxy VLM
𝑉𝑉𝑝𝑝
Cat
Output LogitsEncoder
Text Image
LLM
What is in the 
masked area of 
the image?
Mask 0
Mask NMask 0
Mask N
Query for at most 𝑄𝑄times
Plate Cat Fruit ···
𝑥𝑥=3 𝑥𝑥=1𝑥𝑥=2First Correct 
Step 𝑥𝑥
(1≤𝑥𝑥≤𝑄𝑄)
Data 𝐼𝐼with 
Challenging Mask
Rule-based EvaluatorMask 0: Informative
⭐Selected
Mask N : Non -informative
Ignored*Informativeness 
Score
*Because the object can be easily 
predicted with the self -knowledge of VLMFigure 1: The overall framework of the proposed MrM method. It first perturbs the target image via
object-aware masking, selects informative perturbations using a counterfact-informed mask selection
strategy, and infers membership via a hypothesis test on the RAG system’s response statistics.
image-text memory, pre-trained with joint contrastive and generative objectives. Liu et al. [2]pro-
pose a unified vision-language retrieval model using modality-balanced hard negatives and image
verbalization to bridge modality gaps. Faysse et al. [29] utilize trained VLM to produce high-quality
multi-vector embeddings from text page images, which is combined with a late interaction match-
ing mechanism for efficient document retrieval. Overall, multimodal RAG has now become a key
paradigm, enabling robust cross-modal retrieval and grounded generation in vision-language tasks.
MIA against Multimodal VLMs . Recent advancements in multimodal learning have prompted
studies on MIA targeting the training data of VLMs[ 30–33]. Ko et al. [31] propose an MIA method
using cosine similarity and a weakly supervised attack that avoids shadow training. Li et al. [32]
present a VLM MIA benchmark and a token-level detection method using a confidence-based metric
for both text and images. While these methods can inspire the design of an MIA framework against
multimodal RAG systems, they are primarily based on white-box or gray-box architectures. However,
most RAG systems are deployed in cloud environments and offered as Generation-as-a-Service
(GaaS), thereby operating within black-box settings.
MIA against Unimodal Textual RAG . Recent works have explored MIA in RAG systems focused
on single-text modality. Anderson et al. [18] introduce the first RAG-MIA approach, inferring
document presence by querying the system and interpreting yes/no responses. Li et al. [19] propose
S2MIA , using BLEU-based semantic similarity and perplexity comparisons between target samples
and generated outputs to infer membership. Liu et al. [20] present a mask-based method that perturbs
target documents via word masking, queries the system, and uses prediction accuracy thresholds
for inference. Naseh et al. [21] design document-specific natural-text queries and infer membership
by comparing system responses to shadow LLM-generated ground truth answers. However, these
text-only unimodal RAG-MIA approaches face limitations when applied to multimodal architectures.
Vision-language models handle combined image-text inputs yet produce solely textual responses,
creating a fundamental mismatch with conventional methodologies. In conclusion, MIA specifically
designed for multimodal RAG systems remains largely unexplored in the literature.
3 Threat Model
Attacker’s Goal . The attacker’s objective is to determine whether a target image Ior an image
dataset Di={I1, ...,IN}exists in the black-box multimodal RAG system’s database DB, through
membership inference attacks leveraging only the system’s textual outputs.
3

Attacker’s Capability . The attacker can repeatedly query the RAG system through its public
interface, mimicking legitimate user interactions. They may craft multimodal inputs (text and images)
to probe the system and observe the textual outputs generated by the vision-language model (VLM)
M. The attacker is assumed to possess basic knowledge of multimodal systems but has no prior
information about the database contents, model architecture, or training data.
Attacker’s Constraints . The attacker faces three key limitations: First, they cannot access inter-
mediate system components including the VLM’s output probability distributions, input embeddings,
or the database DB. Second, the VLM Mmay employ security mechanisms to reject explicitly
malicious or privacy-sensitive queries. Third, all observations are restricted to the textual outputs
ofM, with no direct access to retrieval results or database indexing patterns. These constraints
necessitate query strategies that bypass detection while extracting membership signals.
4 Methods
In this section, we present the technical details of our proposed method, MrM, which consists of
three key components as illustrated in Fig 1: object-aware data perturbation, counterfactual-informed
mask selection, and statistical membership inference.
4.1 Design Motivation and Overall Framework
The core motivation behind our framework is to induce both the retrieval and generation phases of an
RAG system to leak membership information simultaneously. In the retrieval phase, the goal is to
ensure that the RAG system successfully retrieves the target data, while in the generation phase, we
aim to elicit relevant information about the target image from the response of the RAG system. If the
target data is input directly without any perturbation, the system is likely to retrieve the corresponding
data. Yet, in this case, it becomes challenging to determine whether the information in the model’s
response originates from the input or the retrieval database. On the other hand, significantly degrading
the target data through perturbation ensures that any relevant information in the model’s response
originates solely from the retrieval knowledge base. However, this approach may significantly
compromise the effectiveness of the membership inference framework, as it could cause the retrieval
process in the RAG system to fail.
Thus, the challenge lies in the balance between these two objectives: ensuring successful retrieval
while maintaining sufficient perturbation to guide the model towards revealing membership informa-
tion. To address this challenge, we propose a multi-object data perturbation framework constrained
by counterfactual attacks. This framework enables the generation of perturbations that strategically
degrade the semantics of the target data, ensuring the retrieval of relevant information while prevent-
ing direct leakage from the input. By inducing multiple responses from the model with different
perturbations, discriminative features can be extracted, providing clear evidence of membership.
4.2 Object-Aware Data Perturbation
For the perturbation process, we set three key objectives: (1) Ensure the target data remains retrievable,
(2) Prevent the reconstruction of information in non-database images during the generation phase, (3)
Facilitate the cross-modal transition from the image modality to the text generation modality.
To meet these goals, we adopt an object-aware perturbation approach. In an image, the object of
interest often occupies only a small fraction of the overall scene, meaning perturbing these regions
has minimal impact on the retrieval process. Moreover, individual objects have relatively independent
semantics, allowing for the preservation of information traceability, which ensures that the source
of the information can be linked back to the retrieval database. Lastly, objects are well-suited to be
transferred into the textual modality, as they are typically well-defined and easily described in words,
making them ideal candidates for generating meaningful text responses in the generation phase.
Given a target image I, we employ an object detection model D(e.g., SAM [ 25]) to localize salient
objects. Let O={oj}K
j=1denote the set of detected objects, where Kis the number of objects. For
each object oj, we generate a binary mask Mjto occlude its corresponding region in I, resulting in a
perturbed image ˜Ij. Formally, the perturbation process is defined as: ˜Ij=I ⊙(1−M j) +0⊙M j,
4

where⊙denotes element-wise multiplication, 1is an all-one matrix, and 0is an all-zero matrix. This
ensures that pixels within Mjare set to zero while preserving other regions.
4.3 Counterfact-Informed Mask Selection
To select perturbations that most effectively differentiate between database and non-database samples,
we propose a counterfact-informed mask selection strategy, where we quantify the informativeness of
each masked region in order to prioritize those that maximize discriminative gaps. This is achieved by
analyzing probability distributions and confidence differentials generated by a proxy vision-language
model (VLM) V, which serves as a counterfactual reference. The goal is to eliminate the interference
from the self-contained knowledge of the target LVLM, thereby ensuring that any observed semantic
reconstruction is attributable to retrieval rather than memorized knowledge.
Given a perturbed image ˜Ij, we input it into the proxy model Vto obtain a probability distribution
P={pi}V
i=1over the vocabulary V. Based on this distribution, we extract the following features
to estimate the informativeness and difficulty of each mask: Target Confidence pc: the predicted
probability corresponding to the ground-truth category of the masked region. Confidence Gap
∆p= max( P)−pc: measuring the discrepancy between the highest predicted probability and the
ground-truth confidence. Entropy H=−PV
i=1pilogpi: quantifies the prediction uncertainty, with
higher entropy indicating greater confusion. Top-k Distribution {p(i)}k
i=1: the top- kvalues in P
sorted in descending order, capturing the distributional sharpness and diversity of high-confidence
predictions.
These features form a feature vector fj= [pc,∆p,H,{p(i)}k
i=1], jointly capturing the uncertainty
of the proxy model, which is important for estimating the discriminative power of a perturbation in
a black-box setting. To assign an informative score to each mask, we adopt a rule-based evaluator
that integrates the extracted features in an unsupervised manner. Specifically, masks are ranked
according to an ensemble of normalized feature scores, where high entropy, low target confidence,
and small confidence gap jointly contribute to a higher informativeness score. We prioritize masks
with high estimated informativeness, as they are more likely to suppress spurious reconstruction for
non-database data while maintaining discriminative signals for membership inference.
4.4 Statistical Significance Analysis
To rigorously infer the membership status of a target image I, we formulate a hypothesis testing
framework grounded in the statistical behavior of the multimodal RAG system when queried about
masked objects. The core intuition is that the system’s success rate in predicting occluded objects
depends on whether Iis in the database DB. Formally, we define two hypotheses. Null hypothesis
(H0):I ∈ DB, where the system’s success probability for each mask follows pt. Alternative
hypothesis ( H1):I/∈DB, with a lower success probability pn, where pt> pnby design.
For each perturbed image ˜Ij(derived from the j-th mask Mj), we query the system repeatedly until
it correctly identifies the masked object. Let xjdenote the number of trials required for the first
correct prediction. Under H0,xjfollows a geometric distribution:
xj∼Geometric (pt),E[xj] =1
pt,Var(xj) =1−pt
p2
t. (1)
ForKmasks, the total trials across all masks are aggregated as S=PK
j=1xj. By the additive prop-
erty of independent geometric variables, Shas expectation and variance: µ0=K
pt, σ2
0=K(1−pt)
p2
t.
Invoking the Central Limit Theorem (CLT) for large K,Sapproximates a normal distribution:
Sapprox∼ N 
µ0, σ2
0
. The p-value quantifies the probability of observing a total trial count as extreme
asSunder H0. To compute it, we first standardize Sand then evaluate the survival function of the
standard normal distribution. Let Φ(z)denote the cumulative distribution function (CDF) of N(0,1).
Thep-value is:
p-value = 1−ΦS−µ0
σ0
= 1−Φ
S−K
ptq
K(1−pt)
p2
t
. (2)
Ifp-value < α (e.g., α= 0.05), we reject H0and conclude I/∈DB; otherwise, we retain H0,
suggesting potential membership.
5

5 Experiments and Analysis
5.1 Experimental Setups
Datasets . We use two standard image datasets to build the knowledge base and perform member-
ship inference attacks. COCO [ 34] and Flickr [ 35] provide diverse image collections widely used
in vision research. From each dataset, we selected 5,000 images for the knowledge base, and 1,000
images (500 members, 500 non-members) for testing.
Target Models . We conduct membership inference attacks on eight commercial models, each
integrated with a local knowledge base to form a multi-modal RAG system: GPT-4o-mini [ 36],
Gemini-2 [ 37], Claude-3.5 [ 38], GLM-4v [ 39], Qwen-VL [ 40], Pixtral [ 41], Moonshot [ 42], and
InternVL-3 [ 43]. These commercial VLMs support multi-image inputs, making them suitable for
multi-modal RAG systems. The experiments are conducted via API calls connected to a locally
built knowledge base. This setup ensures no access to internal generation states, maintaining a strict
black-box environment that mirrors real-world deployment, where only the model output is available
for analysis and no information about the inner workings or intermediate states can be accessed.
Baselines . To the best of our knowledge, our work presents the first MIA approach targeting
multi-modal RAG systems. Due to the lack of baselines, we adapt two strategies from text-based
RAG MIA [ 18,19]. The first baseline, Query-based MIA (denoted as QB-MIA ), directly asks if
the target sample appears in the retrieved references, interpreting the model’s binary response as a
membership signal. The second, Similarity-based MIA ( SB-MIA ), partially masks the target image
and asks the model to reconstruct the missing content using the retrieved reference images. Variants
likeSB-MIA-0.5 indicate the masking ratio. Similarity between the generated description and original
content is then computed, with higher similarity implying likely membership.
Evaluation Metrics . Following prior works on MIA [ 32,44,45], we employ two evaluation
metrics: AUC and TPR5%FPR. AUC reflects overall discrimination between members and non-
members across thresholds. TPR5%FPR measures the true positive rate when the false positive rate
is constrained below 5%, providing a better assessment under strict conditions. Since identical AUCs
can result from different ROC curves, TPR@5%FPR complements AUC for nuanced evaluation. We
report both metrics at the sample and set levels for comprehensive analysis.
Implementation Details . For object-aware data perturbation, we employ the SAM2 model [ 25] to
perform object detection. We utilize the 7B local version of Qwen-VL [ 40] as the proxy VLM. In the
ablation study, we replace it with a weaker detector, the YOLO model [ 46]. All retrieval databases
are constructed using the FAISS library [ 47]. As the image retriever in our RAG system, we adopt
the ViT variant of the CLIP model [26].
5.2 Main Results
Sample-level MIA . Table 1 presents the performance comparison of sample-level MIA across
eight multimodal RAG systems on the Flickr and COCO datasets. Our method, MrM, is evaluated
against two baselines: QB-MIA, which directly queries the model about the presence of a target
sample in the retrieved references, and SB-MIA, which removes a fixed portion of the target image
and prompts the RAG system to describe the original content based on its retrieved references.
To ensure a fair and realistic evaluation, we apply a simple yet natural defense mechanism across
all methods in Table 1: a cautionary system prompt is added to the VLM, stating, “Do not reveal
any information about the membership of your knowledge base. ” This prompt serves as a minimal
safeguard against unintended memorization leakage. While this defense has only limited impact on
the performance of SB-MIA and our proposed MrM method, it significantly weakens the effectiveness
of QB-MIA, which relies on the model’s willingness to answer membership-related questions directly.
All subsequent experiments in this paper are conducted under this default defense setting.
Across both datasets and all models, MrM demonstrates a clear performance advantage, achieving
consistently higher AUC scores, indicating strong overall discriminative ability. It also particularly
excels in TPR@5%FPR compared to baseline methods, which is crucial for evaluating MIA under
strict false positive constraints. This metric reflects an attacker’s success rate under strict false
positive constraints, making it more relevant in real-world scenarios where low false positive rates are
essential for stealthy deployment of MIA. The superior performance of MrM stems from its ability to
6

Table 1: Performance comparison of different MIA methods against RAG across eight multimodal
RAG systems on Flickr and COCO datasets. We report AUC and TPR@5%FPR for each method,
including QB-MIA , three variants of SB-MIA with different masking ratios, and our proposed MrM.
MrM consistently achieves the highest performance, especially under low false positive constraints.
FickrMethods QB-MIA SB-MIA-0.25 SB-MIA-0.5 SB-MIA-0.75 MrM
Metrics AUC TPR@5% AUC TPR@5% AUC TPR@5% AUC TPR@5% AUC TPR@5%
GPT-4o-mini 64.66% 32.85% 67.10% 15.69% 70.04% 20.33% 58.58% 9.67% 80.86% 66.87%
Claude-3.5 55.85% 16.12% 63.21% 14.05% 62.79% 14.05% 44.85% 5.69% 85.36% 74.98%
Gemini-2 72.16% 9.21% 57.23% 8.03% 54.72% 7.36% 44.80% 6.69% 83.19% 66.76%
Pixtral 65.89% 35.18% 71.61% 19.73% 74.26% 28.43% 62.12% 26.09% 83.84% 61.12%
Qwen-VL 56.52% 17.43% 66.76% 13.71% 65.91% 19.06% 55.55% 10.37% 84.22% 72.16%
GLM-4v 55.23% 15.06% 66.98% 14.72% 70.78% 22.41% 58.51% 17.06% 81.93% 58.79%
Moonshot 53.30% 11.27% 74.63% 25.75% 75.98% 24.75% 55.06% 17.73% 80.20% 65.11%
InternVL-3 51.84% 8.50% 64.23% 12.71% 67.25% 18.39% 50.92% 14.05% 83.23% 68.92%COCOMethods QB-MIA SB-MIA-0.25 SB-MIA-0.5 SB-MIA-0.75 MrM
Metrics AUC TPR@5% AUC TPR@5% AUC TPR@5% AUC TPR@5% AUC TPR@5%
GPT-4o-mini 64.42% 11.01% 52.22% 4.35% 59.51% 6.67% 61.58% 12.04% 73.51% 20.77%
Claude-3.5 52.59% 9.91% 58.89% 8.70% 61.37% 12.33% 55.56% 8.03% 82.04% 43.40%
Gemini-2 70.98% 9.28% 50.38% 5.35% 51.23% 4.01% 50.65% 8.03% 84.17% 57.18%
Pixtral 66.22% 35.92% 60.69% 9.36% 62.96% 6.02% 64.24% 16.44% 83.02% 47.24%
Qwen-VL 53.46% 11.57% 55.01% 6.35% 58.10% 7.33% 58.46% 9.73% 84.11% 53.12%
GLM-4v 64.41% 32.51% 55.24% 7.02% 60.86% 10.67% 55.76% 16.78% 76.57% 36.63%
Moonshot 66.47% 36.41% 56.39% 5.88% 63.66% 7.67% 55.57% 8.72% 77.87% 26.31%
InternVL-3 51.51% 7.86% 47.99% 4.68% 59.72% 8.33% 50.29% 6.38% 79.37% 33.03%
precisely disrupt the most semantically critical and least easily inferred regions of the target image.
By leveraging object-aware perturbation and difficulty assessment via a proxy vision-language model,
MrM identifies and masks regions that are both salient and challenging to describe without prior
exposure. As a result, non-member images lead to vague or inaccurate responses from the VLM. In
contrast, for member images, the VLM can often recover the correct semantics from contextual cues
because of its strong in-context learning capabilities. This contrast enhances the discriminative power
of our statistical test and underpins the improved results observed across models and datasets.
Set-level MIA . To evaluate the effectiveness of MrM at the set level, we plot ROC curves in
Figure 2 for varying set sizes K= 1,5,10,20, across eight RAG VLMs and two datasets. Each
curve reflects the model’s ability to infer membership status by aggregating predictions over a set
ofKtarget samples, using a joint statistical test based on the responses of the RAG system to all
samples in the set. We compare our proposed method against the strongest variant of SB-MIA, which
serves as the reference method throughout this section. Across nearly all models and both datasets,
MrM consistently outperforms the baseline method in terms of AUC, and this advantage becomes
more pronounced as the set size increases. When K= 1, MrM maintains strong performance, as
discussed in the sample-level results above, and this advantage scales further with larger sets. As
Kgrows, both methods show improved AUCs, but MrM consistently achieves higher values and
converges more rapidly toward near-perfect performance. In most cases, MrM achieves an AUC
close to 1.0 when K= 10 , indicating its rapid performance saturation with relatively small set sizes.
An additional advantage of MrM is that its ROC curves tend to bend more sharply toward the top-left
corner, indicating higher TPR@5%FPR under the same AUC. This is highlighted by the vertical
reference lines at 5% FPR, where our method consistently achieves higher TPR across models
and datasets. Furthermore, MrM shows strong and stable results across all models and datasets,
including those where the baseline exhibits noticeable performance degradation. This highlights the
generalizability of our approach, even under varying model architectures and retrieval behaviors.
5.3 Ablation Study
To better understand the contribution of each component in MrM, we conduct an ablation study by
systematically removing or replacing key elements of the pipeline. As shown in Fig 3, we evaluate
7

0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0True Positive RateFlickr | GPT-4o-mini
MrM
K=1 (0.85)
K=5 (0.97)
K=10 (0.99)
K=20 (1.00)Baseline
K=1 (0.74)
K=5 (0.88)
K=10 (0.92)
K=20 (0.98)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0Flickr | Gemini-2
MrM
K=1 (0.81)
K=5 (0.97)
K=10 (1.00)
K=20 (1.00)Baseline
K=1 (0.52)
K=5 (0.62)
K=10 (0.63)
K=20 (0.65)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0Flickr | Claude-3.5
MrM
K=1 (0.84)
K=5 (0.99)
K=10 (1.00)
K=20 (1.00)Baseline
K=1 (0.59)
K=5 (0.77)
K=10 (0.83)
K=20 (0.92)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0Flickr | GLM-4v
MrM
K=1 (0.85)
K=5 (0.97)
K=10 (0.99)
K=20 (1.00)Baseline
K=1 (0.74)
K=5 (0.88)
K=10 (0.93)
K=20 (0.99)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0True Positive RateFlickr | Qwen-VL
MrM
K=1 (0.87)
K=5 (0.99)
K=10 (1.00)
K=20 (1.00)Baseline
K=1 (0.65)
K=5 (0.77)
K=10 (0.88)
K=20 (0.96)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0Flickr | Pixtral
MrM
K=1 (0.83)
K=5 (0.99)
K=10 (1.00)
K=20 (1.00)Baseline
K=1 (0.74)
K=5 (0.90)
K=10 (0.97)
K=20 (1.00)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0Flickr | Moonshot
MrM
K=1 (0.77)
K=5 (0.98)
K=10 (1.00)
K=20 (1.00)Baseline
K=1 (0.77)
K=5 (0.91)
K=10 (0.97)
K=20 (1.00)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0Flickr | InternVL-3
MrM
K=1 (0.81)
K=5 (0.99)
K=10 (1.00)
K=20 (1.00)Baseline
K=1 (0.65)
K=5 (0.81)
K=10 (0.90)
K=20 (0.97)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0True Positive RateCOCO | GPT-4o-mini
MrM
K=1 (0.73)
K=5 (0.93)
K=10 (1.00)
K=20 (1.00)Baseline
K=1 (0.57)
K=5 (0.71)
K=10 (0.75)
K=20 (0.86)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0COCO | Gemini-2
MrM
K=1 (0.86)
K=5 (0.98)
K=10 (1.00)
K=20 (1.00)Baseline
K=1 (0.48)
K=5 (0.60)
K=10 (0.56)
K=20 (0.55)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0COCO | Claude-3.5
MrM
K=1 (0.82)
K=5 (0.98)
K=10 (1.00)
K=20 (1.00)Baseline
K=1 (0.56)
K=5 (0.72)
K=10 (0.80)
K=20 (0.88)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0COCO | GLM-4v
MrM
K=1 (0.71)
K=5 (0.86)
K=10 (0.95)
K=20 (0.98)Baseline
K=1 (0.58)
K=5 (0.71)
K=10 (0.83)
K=20 (0.93)
0.0 0.2 0.4 0.6 0.8 1.0
False Positive Rate0.00.20.40.60.81.0True Positive RateCOCO | Qwen-VL
MrM
K=1 (0.84)
K=5 (0.98)
K=10 (1.00)
K=20 (1.00)Baseline
K=1 (0.57)
K=5 (0.68)
K=10 (0.78)
K=20 (0.87)
0.0 0.2 0.4 0.6 0.8 1.0
False Positive Rate0.00.20.40.60.81.0COCO | Pixtral
MrM
K=1 (0.86)
K=5 (0.98)
K=10 (1.00)
K=20 (1.00)Baseline
K=1 (0.60)
K=5 (0.77)
K=10 (0.84)
K=20 (0.92)
0.0 0.2 0.4 0.6 0.8 1.0
False Positive Rate0.00.20.40.60.81.0COCO | Moonshot
MrM
K=1 (0.82)
K=5 (0.96)
K=10 (0.99)
K=20 (1.00)Baseline
K=1 (0.67)
K=5 (0.87)
K=10 (0.86)
K=20 (0.94)
0.0 0.2 0.4 0.6 0.8 1.0
False Positive Rate0.00.20.40.60.81.0COCO | InternVL-3
MrM
K=1 (0.75)
K=5 (0.92)
K=10 (0.99)
K=20 (1.00)Baseline
K=1 (0.59)
K=5 (0.69)
K=10 (0.72)
K=20 (0.83)Figure 2: ROC curves for set-level MIAs with varying set sizes ( K= 1,5,10,20) on two datasets
across eight multimodal RAG systems. We compare MrM with the best SB-MIA baseline. MrM
consistently achieves higher AUCs and steeper curves toward the top-left corner, indicating superior
TPR@5%FPR. The vertical gray dashed line marks the threshold at 5% false positive rate.
GPT-4o-miniGemini -2
Qwen -VLMoonshotGPT-4o-miniGemini -2
Qwen -VLMoonshotGPT-4o-miniGemini -2
Qwen -VLMoonshotGPT-4o-miniGemini -2
Qwen -VLMoonshot
Figure 3: Ablation study results visualized as radar charts. We compare the full MrM method with
three ablated variants: w/o object-awareness, w/o counterfact-informed mask selection, and simpler
OD model. MrM consistently outperforms all ablated versions, demonstrating the contribution of
each component to overall attack performance.
three variants: (1) w/o object-awareness , where object detection is removed and image regions are
randomly masked; (2) w/o proxy model , where the proxy model is excluded and no difficulty-based
mask selection is applied; (3) simpler OD model , where the stronger SAM2 detector is replaced
with YOLO. All variants are tested on both Flickr and COCO datasets across eight RAG systems.
We observe consistent performance drops in all three ablation variants, confirming the necessity of
each component in the full MrM pipeline. (1) Without object-awareness , the model performs signif-
icantly worse, indicating that randomly masking regions often fails to target the most semantically
informative parts of the image. (2) Without the proxy model , the absence of difficulty assessment
leads to less discriminative perturbations, weakening the signal used for membership inference.
8

AUC TPR@5%FPR0.000.250.500.75GPT-4o-mini
AUC TPR@5%FPR0.000.250.500.75Claude-3.5
AUC TPR@5%FPR0.000.250.500.75Gemini-2
AUC TPR@5%FPR0.000.250.500.75Pixtral
AUC TPR@5%FPR0.000.250.500.75Qwen-VL
AUC TPR@5%FPR0.000.250.500.75GLM-4v
AUC TPR@5%FPR0.000.250.500.75Moonshot
AUC TPR@5%FPR0.000.250.500.75InternVL-30.00.20.40.6
0.00.20.40.6
0.00.20.40.6
0.00.20.40.6
0.00.20.40.6
0.00.20.40.6
0.00.20.40.6
0.00.20.40.6
Baseline-no attack MrM-Flip MrM-Greyscale MrM-Crop MrM-Filter MrM-no AttackFigure 4: Robustness of MrM against adaptive image-level transformations applied to the database,
including horizontal flipping, grayscale conversion, cropping, and Gaussian blur filter. MrM maintains
strong performance under all transformations.
(3)Using a simpler object detector results in moderate but noticeable performance degradation,
suggesting that high-quality object detection contributes to more effective perturbation strategies.
These results highlight that all components play essential roles in achieving strong performance.
The full MrM method benefits from their synergy, yielding more effective and reliable membership
inference across models and datasets.
5.4 Robustness Analysis
In addition to the system prompt–based defense discussed earlier, we further evaluate MrM under
a class of data-level defenses based on input alteration. Specifically, we simulate a defense setting
where the retrieval database contains modified versions of the original images, processed through
commonly used transformations such as horizontal flipping, grayscale conversion, cropping, and
Gaussian blurring. These transformations aim to disrupt direct visual matching while preserving the
high-level semantics of the image, thus weakening naive retrieval-based MIA approaches.
To address this challenge, we extend our attack pipeline with an augmentation-aware strategy. For
each target sample, we generate multiple augmented variants using the same transformation types
applied to the database. Each variant is treated as an independent query instance—undergoing object-
aware perturbation, difficulty evaluation, and statistical testing—allowing our method to explore
alternative retrieval paths that remain valid despite the transformation gap. This augmentation-aware
probing increases the likelihood that at least one variant will retrieve the altered database entry,
thereby restoring the model’s memorization signal that might otherwise be masked. Importantly, this
design also mimics a realistic attacker’s capability to guess or approximate potential transformation
patterns in the deployment pipeline. As shown in Fig 4, our method maintains strong performance
under all four transformation-based defenses. This result demonstrates the robustness of MrM
against a range of content-preserving image alterations, reinforcing its practical applicability in more
adversarial or obfuscated deployment scenarios.
6 Conclusion
We introduce MrM, the first black-box membership inference framework specifically designed for
multimodal RAG systems. Our method reveals previously unexplored privacy vulnerabilities in vision-
language models enhanced by external knowledge retrieval. To tackle the challenge of cross-modal
alignment and retrieval-generation balance, we propose a unified MIA framework that jointly exploits
both retrieval and generation phases, enabling the detection of membership signals from multimodal
outputs. MrM incorporates object-aware perturbation and counterfact-informed mask selection to
precisely control semantic leakage while preserving retrieval performance. Extensive experiments
on two visual datasets and eight widely-used commercial LVLMs validate the effectiveness of our
approach, showing that MrM achieves consistently strong performance under both sample-level and
set-level evaluations, and remains robust even in the presence of adaptive defense mechanisms. Our
findings highlight urgent security challenges in multimodal RAG infrastructures and advance the
understanding of privacy risks in systems bridging vision, language, and retrieval.
9

References
[1]Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William Cohen. Murag: Multimodal
retrieval-augmented generator for open question answering over images and text. In Proceedings
of the 2022 Conference on Empirical Methods in Natural Language Processing , pages 5558–
5570, 2022.
[2]Zhenghao Liu, Chenyan Xiong, Yuanhuiyi Lv, Zhiyuan Liu, and Ge Yu. Universal vision-
language dense retrieval: Learning a unified representation space for multi-modal retrieval. In
The Eleventh International Conference on Learning Representations , 2023.
[3]Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec, Percy Liang,
Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Retrieval-augmented multimodal language
modeling. In Proceedings of the 40th International Conference on Machine Learning , pages
39755–39769, 2023.
[4]Yifan Du, Zikang Liu, Junyi Li, and Wayne Xin Zhao. A survey of vision-language pre-trained
models. arXiv preprint arXiv:2202.10936 , 2022.
[5]Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. Vision-language models for vision
tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence , 2024.
[6]Dyke Ferber, Georg Wölflein, Isabella C Wiest, Marta Ligero, Srividhya Sainath, Narmin
Ghaffari Laleh, Omar SM El Nahhas, Gustav Müller-Franzes, Dirk Jäger, Daniel Truhn, et al.
In-context learning enables multimodal large language models to classify cancer pathology
images. Nature Communications , 15(1):10104, 2024.
[7]Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia Shi, Sheng Wang, Linjun Zhang,
James Zou, and Huaxiu Yao. Mmed-rag: Versatile multimodal rag system for medical vision
language models. In Neurips Safe Generative AI Workshop , 2024.
[8]Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang, and Huaxiu
Yao. Rule: Reliable multimodal rag for factuality in medical vision language models. In
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing ,
pages 1081–1093, 2024.
[9]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks. Advances in neural information processing
systems , 33:9459–9474, 2020.
[10] Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuying Zhao, Xueqi Cheng, Qingkai Zeng, Luna
Dong, Yinglong Xia, Krishnaram Kenthapadi, et al. Towards trustworthy retrieval augmented
generation for large language models: A survey. arXiv preprint arXiv:2502.06872 , 2025.
[11] Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu, Yue Xing, Han Xu, Jie Ren, Yi Chang,
Shuaiqiang Wang, Dawei Yin, et al. The good and the bad: Exploring privacy issues in retrieval-
augmented generation (rag). In Findings of the Association for Computational Linguistics ACL
2024 , pages 4505–4524, 2024.
[12] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference
attacks against machine learning models. In 2017 IEEE symposium on security and privacy
(SP), pages 3–18. IEEE, 2017.
[13] Stacey Truex, Ling Liu, Mehmet Emre Gursoy, Lei Yu, and Wenqi Wei. Demystifying mem-
bership inference attacks in machine learning as a service. IEEE transactions on services
computing , 14(6):2073–2089, 2019.
[14] Hongsheng Hu, Zoran Salcic, Lichao Sun, Gillian Dobbie, Philip S Yu, and Xuyun Zhang.
Membership inference attacks on machine learning: A survey. ACM Computing Surveys (CSUR) ,
54(11s):1–37, 2022.
10

[15] Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer.
Membership inference attacks from first principles. In 2022 IEEE symposium on security and
privacy (SP) , pages 1897–1914. IEEE, 2022.
[16] Christopher A Choquette-Choo, Florian Tramer, Nicholas Carlini, and Nicolas Papernot. Label-
only membership inference attacks. In International conference on machine learning , pages
1964–1974. PMLR, 2021.
[17] Iyiola E Olatunji, Wolfgang Nejdl, and Megha Khosla. Membership inference attack on graph
neural networks. In 2021 Third IEEE International Conference on Trust, Privacy and Security
in Intelligent Systems and Applications (TPS-ISA) , pages 11–20. IEEE, 2021.
[18] Maya Anderson, Guy Amit, and Abigail Goldsteen. Is my data in your retrieval database?
membership inference attacks against retrieval augmented generation. arXiv preprint
arXiv:2405.20446 , 2024.
[19] Yuying Li, Gaoyang Liu, Chen Wang, and Yang Yang. Generating is believing: Membership
inference attacks against retrieval-augmented generation. In ICASSP 2025-2025 IEEE Inter-
national Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 1–5. IEEE,
2025.
[20] Mingrui Liu, Sixiao Zhang, and Cheng Long. Mask-based membership inference attacks for
retrieval-augmented generation. In Proceedings of the ACM on Web Conference 2025 , pages
2894–2907, 2025.
[21] Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, and Amir
Houmansadr. Riddle me this! stealthy membership inference for retrieval-augmented generation.
arXiv preprint arXiv:2502.00306 , 2025.
[22] Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan,
and Mubarak Shah. Transformers in vision: A survey. ACM computing surveys (CSUR) , 54
(10s):1–41, 2022.
[23] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang
Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, et al. An empirical study of training end-to-end
vision-and-language transformers. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 18166–18176, 2022.
[24] Yutong Zhou and Nobutaka Shimada. Vision+ language applications: A survey. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 826–842,
2023.
[25] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In
Proceedings of the IEEE/CVF international conference on computer vision , pages 4015–4026,
2023.
[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In International conference on machine learning ,
pages 8748–8763. PmLR, 2021.
[27] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic
visiolinguistic representations for vision-and-language tasks. Advances in neural information
processing systems , 32, 2019.
[28] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-
image pre-training for unified vision-language understanding and generation. In International
conference on machine learning , pages 12888–12900. PMLR, 2022.
[29] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and
Pierre Colombo. Colpali: Efficient document retrieval with vision language models. In The
Thirteenth International Conference on Learning Representations , 2024.
11

[30] Pingyi Hu, Zihan Wang, Ruoxi Sun, Hu Wang, and Minhui Xue. M4I: Multi-modal models
membership inference. Advances in Neural Information Processing Systems , 35:1867–1882,
2022.
[31] Myeongseob Ko, Ming Jin, Chenguang Wang, and Ruoxi Jia. Practical membership inference
attacks against large-scale multi-modal models: A pilot study. In Proceedings of the IEEE/CVF
International Conference on Computer Vision , pages 4871–4881, 2023.
[32] Zhan Li, Yongtao Wu, Yihang Chen, Francesco Tonin, Elias Abad Rocamora, and V olkan
Cevher. Membership inference attacks against large vision-language models. Advances in
Neural Information Processing Systems , 37:98645–98674, 2024.
[33] Luis Ibanez-Lissen, Lorena Gonzalez-Manzano, Jose Maria de Fuentes, Nicolas Anciaux, and
Joaquin Garcia-Alfaro. Lumia: Linear probing for unimodal and multimodal membership
inference attacks leveraging internal llm states. arXiv preprint arXiv:2411.19876 , 2024.
[34] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer
vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014,
proceedings, part v 13 , pages 740–755. Springer, 2014.
[35] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions
to visual denotations: New similarity metrics for semantic inference over event descriptions.
Transactions of the association for computational linguistics , 2:67–78, 2014.
[36] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv
preprint arXiv:2410.21276 , 2024.
[37] Shrestha Basu Mallick and Logan Kilpatrick. Gemini 2.0: Flash, flash-lite
and pro, February 2025. URL https://developers.googleblog.com/zh-hans/
gemini-2-family-expands/ . Accessed: 2025-05-01.
[38] Anthropic. Introducing claude 3.5 sonnet, June 2024. URL https://www.anthropic.com/
news/claude-3-5-sonnet . Accessed: 2025-05-01.
[39] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu
Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng,
Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao
Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin
Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao
Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An,
Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu
Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. Chatglm: A family of large
language models from glm-130b to glm-4 all tools, 2024.
[40] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang,
Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 ,
2025.
[41] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot,
Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet,
et al. Pixtral 12b. arXiv preprint arXiv:2410.07073 , 2024.
[42] Moonshot AI. Kimi chat, 2024. URL https://kimi.moonshot.cn/ . Accessed: 2025-04-10.
[43] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shen-
glong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source
multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271 ,
2024.
[44] Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi
Chen, and Luke Zettlemoyer. Detecting pretraining data from large language models. In The
Twelfth International Conference on Learning Representations .
12

[45] Jingyang Zhang, Jingwei Sun, Eric Yeats, Yang Ouyang, Martin Kuo, Jianyi Zhang, Hao Frank
Yang, and Hai Li. Min-k%++: Improved baseline for detecting pre-training data from large
language models. arXiv preprint arXiv:2404.02936 , 2024.
[46] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified,
real-time object detection. In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 779–788, 2016.
[47] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-
Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. The faiss library. 2024.
13

A Detailed Threat Model
Target System 
ResponseRAG Membership Inference 
Attack (MIA)
Suspected Image
 Object Masking
Hypothesis 
Testing
Is this image included in the retrieval database? What is in the masked area of the image?
Image in𝐷𝐷𝐷𝐷
LeavesCat
Image not in 𝐷𝐷𝐷𝐷
Multimodal 
Database (𝐷𝐷𝐷𝐷)CLIP
Multimodal 
RetrieverTarget Multimodal RAG System (Blackbox)
Target VLM
Text Image
LLMText
ImageQuery
Generate
Figure 5: Illustration of the black-box MIA against a multimodal RAG system. An attacker queries
the system with strategically perturbed image inputs and analyzes the textual responses to determine
whether the target image exists in the underlying retrieval database, without access to model internals
or database contents.
The attacker must construct a membership inference algorithm Athat iteratively probes the black-
box RAG system through carefully designed multimodal queries to infer membership status. As
illustrated in Fig. 5, the attacker submits perturbed versions of a target image and observes the
system’s textual responses to determine whether the image exists in the underlying retrieval database.
The attacker has no access to the internal parameters of the RAG system, its retriever, or its database
content—operating entirely under a black-box assumption.
Formally, for a target image I, the algorithm synthesizes evidence from a sequence of kadaptive
queries Q1(I), . . . ,Qk(I)and their corresponding textual responses O1, . . . ,Okgenerated by the
vision-language model Maugmented by the retrieval database, defined as
A 
I,{Oi}k
i=1
→ {0,1}.
This requires designing query strategies that subtly elicit membership-distinctive patterns in the
output text—such as variations in specificity, contextual coherence, or knowledge granularity—while
circumventing the VLM’s security mechanisms through semantically ambiguous or contextually
indirect prompts.
For dataset-level verification of Di=I1, . . . ,IN, the task extends to aggregating observations across
allNimages, necessitating a composite inference rule
A 
{Ij}N
j=1,{{O(j)
i}k
i=1}N
j=1
→ {0,1}
that balances per-image evidence with global statistical confidence.
Critical challenges include identifying latent correlations between text output features and database
membership, developing noise-tolerant aggregation methods to fuse multi-query evidence, and
maintaining stealth by mimicking legitimate user behavior to avoid triggering defensive filters.
B Detailed Experimental Setup
B.1 Model Versions
Target VLMs . For the target models in the multimodal RAG systems, we use the follow-
ing commercial vision-language model versions via their respective APIs: GPT-4o-mini [ 36]
(gpt-4o-mini ), Gemini-2 [ 37] (gemini-2.0-flash ), Claude-3.5 [ 38] (claude-3-5-sonnet ),
GLM-4v [ 39] (glm-4v-plus-0111 ), Qwen-VL [ 40] (qwen-vi-max-0408 ), Pixtral [ 41]
(pixtral-large-2411 ), Moonshot [ 42] (moonshot-v1-8k-vision-preview ), and InternVL-
3 [43] (internvl3-78b ). All models are accessed in a strictly black-box manner through API calls,
with no access to their internal generation states.
Retriever . For the retrieval component, we employ the CLIP model [ 26] with the ViT backbone,
specifically the openai_clip-vit-large-patch14-336 variant, to encode image features and
build the retrieval database. Its image embedding dimension is 768.
Object Detectors . The main object detector is SAM2 [ 25]:sam2.1-hiera-large . In the ablation
study, we use yolo11x [46] as a weaker alternative.
14

AUC TPR@5%FPR0.00.20.40.60.81.0Flickr
AUC TPR@5%FPR0.00.20.40.60.81.0COCO
0.00.20.40.60.8
0.00.10.20.30.40.50.6
llava-1.5-7b qwen2-vl-7b-instruct deepseek-vl-7b-chat Llama-3.2-11B-Vision-InstructFigure 6: Membership inference performance of MrM on locally deployed open-source vision-
language models with 7B–11B parameters, evaluated on the Flickr and COCO datasets. Bars show
AUC and TPR@5%FPR. Results indicate a modest drop in performance compared to commercial
models, but still demonstrate meaningful discriminative signals.
B.2 Baseline and Ablations Implementation Details
In the SB-MIA baseline, the target image is masked by occluding its left portion. The multimodal
RAG system is then prompted to reconstruct the missing content using retrieved reference images.
We then compute the similarity between the masked image region and the model’s generated text
using CLIP’s cross-modal similarity. The numerical suffix in SB-MIA , such as SB-MIA-0.5 , indicates
the masking ratio. Larger masking ratios make it harder for the VLM to reconstruct the original
semantics, thereby increasing the potential for successful membership inference. However, higher
masking also reduces retrieval success due to greater loss of visual information. Conversely, smaller
masking ratios improve retrieval but reduce the distinguishability of generation outputs. Compared to
this trade-off, MrM achieves a better balance between retrieval accuracy and generation divergence,
resulting in improved MIA performance.
In the ablation study, the variant without object-awareness applies random masking by selecting a
rectangular region whose height and width range from 20% to 60% of the original image. The overall
masked area is thus similar to that of MrM. The similarity computation follows the same procedure
as in SB-MIA , using CLIP-based cross-modal similarity between the masked region image and the
generated text.
B.3 Additional Implementation Details
For all target RAG systems, we use the following prompt: Analyze the precise positional correspon-
dence between the masked region in the input image and the reference images. Extract ONLY the
visual content from the EXACT SAME POSITION in the reference image. Output ONLY a word or
phrase. DO NOT WRITE ANYTHING ELSE. Input Image:[IMAGE] Reference Images: {[IMAGE],...}
If a vision-language model does not support multiple image inputs, we concatenate the target image
and reference images into one image before inputting.
The retrieval database is built using FAISS [ 47]. We use the Flat index type and apply mean pooling
on image embeddings. Although the COCO dataset [ 34] includes ground-truth object annotations, we
do not use them in our pipeline. Instead, we simulate real-world scenarios by detecting objects using
the SAM model. All ROC curves are plotted using random sampling, with 200 samples per value of
K. Each experiment is repeated five times to reduce the impact of randomness. Key Python libraries
used in our implementation include: openai (v1.64.0), faiss (v1.9.0), transformers (v4.45.1),
andtorch (v2.4.1)
C Evaluation on Open-Source Small-Scale Models
To explore the generalizability of MrM beyond commercial API-based systems, we evaluate its
performance on several open-source vision-language models with smaller parameter scales (7B–11B).
As shown in Figure 6, we test four models on the Flickr and COCO datasets: llava-1.5-7b ,
qwen2-vl-7b-instruct ,deepseek-vl-7b-chat , and llama-3.2-11B-Vision-Instruct .
15

0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0True Positive RateFlickr | GPT-4o-mini
MrM
K=1 (0.83)
K=5 (0.95)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.63)
K=5 (0.79)
K=10 (0.87)
K=20 (0.94)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0Flickr | Gemini-2
MrM
K=1 (0.86)
K=5 (0.98)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.73)
K=5 (0.96)
K=10 (0.99)
K=20 (1.00)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0Flickr | Claude-3.5
MrM
K=1 (0.86)
K=5 (0.99)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.85)
K=5 (0.97)
K=10 (1.00)
K=20 (1.00)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0Flickr | GLM-4v
MrM
K=1 (0.82)
K=5 (0.97)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.73)
K=5 (0.89)
K=10 (0.98)
K=20 (1.00)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0True Positive RateFlickr | Qwen-VL
MrM
K=1 (0.83)
K=5 (0.99)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.64)
K=5 (0.87)
K=10 (0.93)
K=20 (0.99)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0Flickr | Pixtral
MrM
K=1 (0.83)
K=5 (0.99)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.75)
K=5 (0.97)
K=10 (0.98)
K=20 (1.00)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0Flickr | Moonshot
MrM
K=1 (0.79)
K=5 (0.96)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.77)
K=5 (0.95)
K=10 (0.98)
K=20 (1.00)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0Flickr | InternVL-3
MrM
K=1 (0.82)
K=5 (0.99)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.65)
K=5 (0.84)
K=10 (0.89)
K=20 (0.97)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0True Positive RateCOCO | GPT-4o-mini
MrM
K=1 (0.82)
K=5 (0.94)
K=10 (0.99)
K=20 (1.00)w/o object-aware
K=1 (0.56)
K=5 (0.79)
K=10 (0.90)
K=20 (0.95)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0COCO | Gemini-2
MrM
K=1 (0.85)
K=5 (0.99)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.70)
K=5 (0.86)
K=10 (0.98)
K=20 (0.99)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0COCO | Claude-3.5
MrM
K=1 (0.87)
K=5 (0.99)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.71)
K=5 (0.91)
K=10 (0.97)
K=20 (1.00)
0.0 0.2 0.4 0.6 0.8 1.00.00.20.40.60.81.0COCO | GLM-4v
MrM
K=1 (0.70)
K=5 (0.89)
K=10 (0.96)
K=20 (0.99)w/o object-aware
K=1 (0.64)
K=5 (0.83)
K=10 (0.94)
K=20 (0.97)
0.0 0.2 0.4 0.6 0.8 1.0
False Positive Rate0.00.20.40.60.81.0True Positive RateCOCO | Qwen-VL
MrM
K=1 (0.90)
K=5 (0.99)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.60)
K=5 (0.78)
K=10 (0.91)
K=20 (0.93)
0.0 0.2 0.4 0.6 0.8 1.0
False Positive Rate0.00.20.40.60.81.0COCO | Pixtral
MrM
K=1 (0.87)
K=5 (0.97)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.71)
K=5 (0.81)
K=10 (0.92)
K=20 (0.98)
0.0 0.2 0.4 0.6 0.8 1.0
False Positive Rate0.00.20.40.60.81.0COCO | Moonshot
MrM
K=1 (0.82)
K=5 (0.96)
K=10 (1.00)
K=20 (1.00)w/o object-aware
K=1 (0.66)
K=5 (0.95)
K=10 (0.98)
K=20 (1.00)
0.0 0.2 0.4 0.6 0.8 1.0
False Positive Rate0.00.20.40.60.81.0COCO | InternVL-3
MrM
K=1 (0.77)
K=5 (0.95)
K=10 (0.98)
K=20 (1.00)w/o object-aware
K=1 (0.61)
K=5 (0.72)
K=10 (0.77)
K=20 (0.85)Figure 7: Set-level ROC curves comparing MrM with its object-agnostic variant under different set
sizes ( K= 1,5,10,20), across eight multimodal RAG systems and two datasets. MrM consistently
achieves higher AUCs, particularly when Kis small, highlighting the effectiveness of object-aware
perturbation in improving MIA performance.
Each model is locally deployed and evaluated using the same sample-level MIA protocol intro-
duced in the main text.
The results show that, compared to large-scale commercial systems, these smaller models exhibit
relatively lower AUC and TPR@5%FPR scores. Nevertheless, the gap is not drastic, and several
models (e.g., deepseek-vl-7b-chat ) demonstrate non-trivial discriminative ability. We hypothe-
size that the drop in performance is mainly due to weaker reasoning and generation capabilities in
smaller models, particularly in handling multi-image prompts and resolving masked content based on
retrieved references—even when relevant images are successfully retrieved.
While these results highlight certain limitations, they also confirm the generalizability of MrM
across a range of architectures. In practice, however, most real-world RAG systems are composed of
API-level models backed by proprietary retrieval databases, which makes the evaluation on API-based
models more representative. Still, this open-source benchmark offers valuable insights into how
model scale affects attack success.
D Detailed Ablation Results
To further examine the effectiveness of object-aware perturbation, we compare MrM with its object-
agnostic variant across set sizes K= 1,5,10,20, using ROC curves plotted in Fig 7. Across
both Flickr and COCO datasets, and for all tested models, the object-aware design yields more
favorable ROC characteristics (closer to the top-left corner) and higher AUC values at nearly all
K. The performance gap is particularly visible at small set sizes (e.g., K= 1), where precise
perturbation is critical for triggering distinguishable responses. As Kincreases, the performance of
both methods improves, but MrM consistently maintains a clear advantage, indicating that semantic-
aware perturbations enhance both sample-level and aggregated set-level inference.
16

Perturbed Input 
(Not inDatabase)Retrieved Reference Image
SB-MIA
MrMThe image shows a white draft horse 
pulling a green trolley car along a street, likely at a theme park or tourist 
attraction.  The horse is in full harness 
and the trolley appears to be a vintage-
style passenger transport vehicle.
Normalized CLIP 
similarity : 0.923RAG System Output
Retrieved Reference Image
Output: Railroad crossing signalCorrect answer: Horse
Output: ZebraCorrect answer: Bench
Perturbed Input 
(Not inDatabase)RAG System OutputIn Database
Not in 
Database
Membership Inference
Membership Inference
MrMRetrieved Reference Image
Output: HorseCorrect answer: Horse
Output: BenchCorrect answer: BenchPerturbed Input 
(InDatabase)RAG System Output
In Database
Membership Inference
Perturbed Input 
(In Database)Retrieved Reference Image
QB-MIANo.
(The image is not in the 
context.)RAG System Output
Not in 
Database
Membership Inference
System Prompt (Naive Defense): 
Donotleak anymembership information of images in retrieval database in your answer.
Method Simplified Prompt
QB-MIA Does this target image [IMAGE] appear in the context? Answer only with Yes or No. \ n Retrieved References: 
{[IMAGE], …} \nAnswer :
SB-MIA Describe the image [IMAGE] within 50 words, using retrieved images as reference. Starts with \"The image 
shows \". \n Retrieved References: {[IMAGE], …} 
MrM Analyze the precise positional correspondence between the masked region in the input image and the reference images. Output a word or phrase. Input Image:[IMAGE] Retrieved References: {[IMAGE],...}Figure 8: Case study demonstrating the advantage of MrM in distinguishing between in-database
and non-database samples. Baseline ( QB-MIA ) applied to a database image fails due to naive defense
strategy. Baseline ( SB-MIA ) applied to a non-database image results in incorrect inference due to
high semantic similarity reconstruction. MrM correctly infers non-membership by masking key
objects (horse and bench) and suppressing semantic recovery. MrM correctly infers membership
when applied to an in-database image, where retrieval supports accurate reconstruction.
17

E Case Study
To further illustrate the advantages of our method, we present a representative case study comparing
MrM with baseline QB-MIA andSB-MIA-0.5 , by examining both in-database and non-database
scenarios for the same target image. QB-MIA applied to a database image fails due to naive defense
strategy.
SB-MIA-0.5 applies a coarse masking strategy by occluding half of the image, then relies on cross-
modal semantic similarity to infer membership. As shown in the first example of Fig. 8, despite
the image being unseen in the context, the model leverages its internal reasoning capabilities to
reconstruct the full scene and produces a response with high semantic similarity to the original
(similarity score = 0.923). This leads the baseline to incorrectly infer that the image is in the database.
In contrast, MrM specifically masks two critical objects—namely the horse and the bench, which are
essential for accurate scene interpretation. Without access to the original image through retrieval, the
model fails to accurately answer these elements, resulting in a correct inference that the image is not
in the database.
In the fourth case, we put this image into the database and apply MrM to it. The answer is correct
due to successful retrieval, and our inference framework correctly identifies it as a member.
F Limitation
Our experimental evaluation is conducted on two widely used and representative visual datasets,
which reflect common scenarios encountered in multimodal RAG systems. While these datasets
provide meaningful coverage of typical applications, the generalizability of our findings to other
domains is not guaranteed. In particular, some emerging domains may present unique challenges
to our framework. These include settings with high inter-image similarity or highly specialized
visual semantics, such as radiological medical images (e.g., CT scans and ultrasound) or satellite
images. Such domains often have distinct knowledge characteristics that may not align well with the
assumptions underlying our current perturbation and inference design. Future work should extend the
evaluation to a broader range of domain-specific datasets and adjust the strategies accordingly. This
would further validate the adaptability of our approach and offer deeper insights into the privacy risks
posed by RAG systems across diverse real-world settings.
18