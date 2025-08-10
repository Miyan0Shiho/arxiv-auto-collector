# RAIDX: A Retrieval-Augmented Generation and GRPO Reinforcement Learning Framework for Explainable Deepfake Detection

**Authors**: Tianxiao Li, Zhenglin Huang, Haiquan Wen, Yiwei He, Shuchang Lyu, Baoyuan Wu, Guangliang Cheng

**Published**: 2025-08-06 15:08:16

**PDF URL**: [http://arxiv.org/pdf/2508.04524v1](http://arxiv.org/pdf/2508.04524v1)

## Abstract
The rapid advancement of AI-generation models has enabled the creation of
hyperrealistic imagery, posing ethical risks through widespread misinformation.
Current deepfake detection methods, categorized as face specific detectors or
general AI-generated detectors, lack transparency by framing detection as a
classification task without explaining decisions. While several LLM-based
approaches offer explainability, they suffer from coarse-grained analyses and
dependency on labor-intensive annotations. This paper introduces RAIDX
(Retrieval-Augmented Image Deepfake Detection and Explainability), a novel
deepfake detection framework integrating Retrieval-Augmented Generation (RAG)
and Group Relative Policy Optimization (GRPO) to enhance detection accuracy and
decision explainability. Specifically, RAIDX leverages RAG to incorporate
external knowledge for improved detection accuracy and employs GRPO to
autonomously generate fine-grained textual explanations and saliency maps,
eliminating the need for extensive manual annotations. Experiments on multiple
benchmarks demonstrate RAIDX's effectiveness in identifying real or fake, and
providing interpretable rationales in both textual descriptions and saliency
maps, achieving state-of-the-art detection performance while advancing
transparency in deepfake identification. RAIDX represents the first unified
framework to synergize RAG and GRPO, addressing critical gaps in accuracy and
explainability. Our code and models will be publicly available.

## Full Text


<!-- PDF content starts -->

RAIDX: A Retrieval-Augmented Generation and GRPO
Reinforcement Learning Framework for Explainable Deepfake
Detection
Tianxiao Li
University of Liverpool
Liverpool, United Kingdom
sgtli18@liverpool.ac.ukZhenglin Huang
University of Liverpool
Liverpool, United Kingdom
zhenglin@liverpool.ac.ukHaiquan Wen
University of Liverpool
Liverpool, United Kingdom
sghwen4@liverpool.ac.uk
Yiwei He
University of Liverpool
Liverpool, United Kingdom
yiwei.he@liverpool.ac.ukShuchang Lyu
Beihang University
Beijing, China
lyushuchang@buaa.edu.cnBaoyuan Wu
The Chinese University of Hong Kong
Shenzhen, China
wubaoyuan1987@gmail.com
Guangliang Cheng∗
University of Liverpool
Liverpool, United Kingdom
guangliang.cheng@liverpool.ac.uk
ABSTRACT
The rapid advancement of AI-generation models has enabled the
creation of hyperrealistic imagery, posing ethical risks through
widespread misinformation. Current deepfake detection methods,
categorized as face-specific detectors or general AI-generated de-
tectors, lack transparency by framing detection as a classification
task without explaining decisions. While several LLM-based ap-
proaches offer explainability, they suffer from coarse-grained anal-
yses and dependency on labor-intensive annotations. This paper
introduces RAIDX (Retrieval- Augmented Image Deepfake Detec-
tion and E Xplainability), a novel deepfake detection framework
integrating Retrieval-Augmented Generation (RAG) and Group Rel-
ative Policy Optimization (GRPO) to enhance detection accuracy
and decision explainability. Specifically, RAIDX leverages RAG to
incorporate external knowledge for improved detection accuracy
and employs GRPO to autonomously generate fine-grained textual
explanations and saliency maps, eliminating the need for exten-
sive manual annotations. Experiments on multiple benchmarks
demonstrate RAIDX’s effectiveness in identifying real or fake, and
providing interpretable rationales in both textual descriptions and
saliency maps, achieving state-of-the-art detection performance
while advancing transparency in deepfake identification. RAIDX
represents the first unified framework to synergize RAG and GRPO,
addressing critical gaps in accuracy and explainability. Our code
and models will be publicly available.
CCS CONCEPTS
•Computing methodologies →Computer vision ;Reinforce-
ment learning ; Natural language generation; •Security and
∗Corresponding author.
MM ’25, October 27–31, 2025, Dublin, Ireland
2025. ACM ISBN 979-8-4007-2035-2/2025/10. . . $15.00
https://doi.org/10.1145/3746027.3754798privacy→Human and societal aspects of security and privacy ;
•Human-centered computing →Heat maps.
KEYWORDS
deepfake detection, vision-language model, retrieval-augmented
generation, reinforcement learning, multimodal explainability
ACM Reference Format:
Tianxiao Li, Zhenglin Huang, Haiquan Wen, Yiwei He, Shuchang Lyu,
Baoyuan Wu, and Guangliang Cheng. 2025. RAIDX: A Retrieval-Augmented
Generation and GRPO Reinforcement Learning Framework for Explainable
Deepfake Detection. In Proceedings of the 33rd ACM International Conference
on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland. ACM, New
York, NY, USA, 10 pages. https://doi.org/10.1145/3746027.3754798
1 INTRODUCTION
Segmentation
MaskForgery Mask
Saliency Map
Fine-grained  
Explanation
EncoderEncoder
VLMVLM
RAG(a) Binary Classification (b) Mask -based Detection
(c) Explainable Detector (d) RAIDXPromptImageImage Image
Image
PromptReal Fake
Real FakeReal Fake
Real Fake
Explanation
Figure 1: The RAIDX achieves detection, localization, and
GRPO-enhanced explanation, all without requiring any su-
pervision.
The rapid evolution of AI-generated visual content methods [ 62,
80] has positioned this field as one of the most transformative fron-
tiers in computer vision. Driven by breakthroughs in generativearXiv:2508.04524v1  [cs.CV]  6 Aug 2025

MM ’25, October 27–31, 2025, Dublin, Ireland Li et al.
modeling techniques—such as Generative Adversarial Networks
(GANs) [ 22,31], diffusion models [ 11,56], and Vision-Language
Models (VLMs) [ 40,53,59]—these technologies now produce hy-
perrealistic imagery indistinguishable from authentic photographs.
While such advancements unlock creative potential, they also intro-
duce profound ethical risks, enabling malicious actors to organize
misinformation campaigns and destabilize trust in digital media.
This duality underscores an urgent need for robust and explainable
deepfake detection systems capable of distinguishing synthetic con-
tent from genuine imagery, as well as explaining their judgment
basis.
Current detection methodologies primarily fall into two cat-
egories: face-specific detectors [ 32,78,82], which target facial
manipulation artifacts, and general AI-generated content detec-
tors [ 19,64,65], designed for broader synthetic media identifi-
cation. Despite progress, these approaches predominantly frame
detection as a binary or multi-category classification task (shown
in Figure 1 (a)), offering no transparency into the rationale behind
their decisions. This black-box nature limits practical utility, as
users cannot verify why an image is flagged as fake. To address
this, some mask-based detection methods [ 17,23,43] attempt to
simultaneously provide corresponding mask predictions while per-
forming classification (shown in Figure 1 (b)). In recent years, with
the rapid development of large language models (LLMs), although
numerous LLM-based approaches [ 27,30] strive to provide explain-
ability (shown in Figure 1 (c)), they remain constrained by two
fundamental limitations: (1) coarse-grained analyses that fail
to pinpoint specific manipulation indicators , and (2) depen-
dency on labor-intensive annotations (e.g., mask labeling or
textual descriptions), hindering real-world applicability.
Addressing these limitations requires rethinking both how expla-
nations are generated and what evidence underpins detection. In-
novations in LLM optimization, particularly reinforcement learning
techniques like GRPO [ 14], offer a promising pathway. GRPO en-
hances reasoning by optimizing grouped outputs without auxiliary
critic models, demonstrating success in diverse domains [ 13,26].
Yet, its potential for deepfake detection remains unexplored. Ad-
ditionally, in LLM-based approaches, RAG effectively integrates
external knowledge to boost accuracy and produce precise text
descriptions. Yet, its potential in VLMs or deepfake detection task
remains underexplored.
We propose the Retrieval- Augmented Image Deepfake Detec-
tion and E Xplainability ( RAIDX ), a novel framework that syner-
gizes Retrieval-Augmented Generation RAG [ 35] with GRPO to
achieve state-of-the-art detection accuracy while producing inter-
pretable, fine-grained explanations (shown in Figure 1 (d)). Specifi-
cally, RAIDX utilizes RAG [ 35] to establish a retrieval repository
to boost detection accuracy, while implementing GRPO to uncover
interpretable rationales for model decisions, eliminating reliance on
laborious manual annotation. To our knowledge, RAIDX represents
the first framework to integrate both RAG and GRPO for deepfake
detection, achieving enhanced detection accuracy while providing
fine-grained explanations of its decision-making process.
The main contributions of this paper are as follows:•We introduce RAIDX, a novel deepfake detection framework
to harness both RAG and the GRPO algorithm to enhance
both detection accuracy and decision explainability.
•We are the first to integrate RAG into deepfake detection, en-
hancing detection accuracy by incorporating external knowl-
edge.
•A novel GRPO-driven reinforcement learning framework
is proposed to automatically generate fine-grained textual
descriptions and precise saliency maps, localizing suspected
fake regions and providing interpretable explanations with-
out manual annotations.
•Extensive experiments on multiple deepfake detection bench-
marks demonstrate that RAIDX effectively identifies and
delineates suspected fake regions within images.
2 RELATED WORK
2.1 Vision-Language Models
Vision-language models (VLMs) have emerged as powerful tools
for multimodal tasks, including deepfake detection. The rapid ad-
vancement of large language models (LLMs) such as GPT-4 [ 1] and
LLaMA [ 66] has driven the evolution of VLMs, enabling seamless
integration of vision and language modalities. Early approaches
to VLMs, such as CLIP [ 52], utilized contrastive learning to align
vision and language representations in a shared embedding space.
Subsequent models like BLIP [ 37] introduced more advanced pre-
training strategies, incorporating retrieval-augmented learning and
cross-attention mechanisms to enhance contextual understanding.
Recent advancements, such as LLaVA [ 40] and Qwen2-VL [ 68],
have integrated ViT with LLMs, enabling vision-conditioned text
generation and multimodal reasoning. Building on these founda-
tions, models like Qwen2.5-Omni [ 77] introduce novel architectures
and techniques to handle complex multimodal tasks, such as video
dialogue and reasoning, while maintaining strong performance in
visual and linguistic understanding.
2.2 Deepfake Detection Methods
Image deepfake detection has evolved significantly with advance-
ments in AI technologies. Traditional methods [ 6,9,67], primarily
based on Convolutional Neural Networks (CNNs), focus on identi-
fying pixel-level anomalies such as unnatural textures and lighting
inconsistencies. These Non-VLM-based approaches, exemplified by
models like XceptionNet [ 10] and MesoNet [ 2], have shown high
precision in specific scenarios but often struggle with generalization
due to their reliance on dataset-specific features.
More recently, the integration of VLMs has introduced a new
paradigm in deepfake detection. VLM-based methods leverage the
reasoning and cross-modal capabilities of these models to provide
more flexible and interpretable detection. Models such as Antifake-
Prompt [ 7], ForgeryGPT [ 42], ForgeryTalker [ 38], ForgerySleuth
[63], SIDA [ 27] and So-Fake-R1 [ 28] utilize the capabilities of VLM
to enhance detection accuracy and provide robust support for lo-
calization and explanation tasks. In the video domain, BusterX [ 74]
and BusterX++ [ 75] are outstanding representatives. Most of these
methods rely heavily on extensive manual annotation of masks or
textual explanatory descriptions. Additionally, the interpretable de-
scriptions learned by current models tend to be coarse. In contrast,

RAIDX: A RAG-GRPO Framework for Explainable Deepfake Detection MM ’25, October 27–31, 2025, Dublin, Ireland
ViT
Isthis image real?
Prompt EngineeringPlease carefully analyze this image 
and provide a detailed explanation.
LLM LoRA(GRPO Training)
<think>
The image depicts a cowboy on horseback chasing a steer through a dusty environment, likely during a rodeo or cattle drive sc ene . The lighting 
suggests a sunset or sunrise, casting a warm glow over the scene. The dust kicked up by the animals adds to the realism of th e setting. \n\n### 
Analysis: \n1. **Lighting and Shadows**: While the warm directional lighting appears consistent, subtle inconsistencies are visib le around the 
horse‘s hooves and the steer’s legs, where shadows and contact points seem unnaturally soft or floating, suggesting an imprecise simulation of 
ground interaction.  \n2. **Anatomical Irregularities **: The hat shows unnatural curvature and blending with the background light , especially 
around the edges. The face lacks realistic texture, and fine facial details . The horse‘s face appears slightly stylized, with a glossy finish and 
disproportionately smooth edges around the nostrils and eyes . The hooves, do not deform the dust or ground as expected and cast shadows that 
appear artificially rendered ……\n\n### Conclusion:
Despite its strong visual coherence, this image reveals several key synthetic artifacts —including smooth facial features, anatom ically 
implausible hooves and tails, and overly uniform dust patterns. These features strongly indicate that the image is AI -generated rather than a real 
photographic capture
</think><answer>FAKE</answer>RAGTokenizerRAIDX
Trainable
 Frozen
Figure 2: Framework of RAIDX: It integrates retrieval-augmented prompting with GRPO-enhanced reasoning for deepfake
detection. A ViT and RAG module construct informative prompts using similar images, which are processed by a partially
trainable LLM to output saliency-guided localization and fine-grained explanations without requiring any supervision.
RAIDX relies on the GRPO reinforcement learning strategy, which
can produce relatively accurate saliency maps and fine-grained
textual explanation outputs without requiring any manual text or
mask annotations.
2.3 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) [ 21] is a general paradigm
which enhances LLMs by including relevant information retrieved
from external databases into the input. RAG typically consists of
three phases: indexing ,retrieval , and generation .Indexing con-
structs external databases and their retrieval index from external
data sources. Retrieval utilizes these indexes to fetch the relevant
document chunks as context, given a user query. Generation inte-
grates the retrieved context into the input prompt for LLMs, and
LLMs then generate the final output based on the augmented in-
puts. RAG has been widely used in various domains [ 34,61,81].
Although RAG has achieved many successful cases in LLM tasks,
there are few examples [ 45,51] of its integration with VLMs. To
the best of our knowledge, RAIDX is the first to apply RAG in the
deepfake detection task.
2.4 Reinforcement Learning in VLMs
The integration of reinforcement learning (RL) into multimodal sys-
tems has emerged as a promising direction, particularly for tasks
requiring alignment between vision and language. Recent advances
in techniques such as Proximal Policy Optimization (PPO) [ 57]
and Group Relative Policy Optimization (GRPO) [ 15] have demon-
strated their potential to enhance reasoning quality and factual
correctness in language models, often by optimizing model outputsthrough structured feedback. GRPO, in particular, has gained signif-
icant attention due to its effectiveness in improving the reasoning
capabilities of LLMs by leveraging rule-based rewards. In the vi-
sual domain, the application of RL to VLMs has shown substantial
promise [ 8,26,60]. Experimental results from the VLM-R1 [ 60]
demonstrate that RL-based models not only deliver competitive
performance on visual understanding tasks but also surpass super-
vised fine-tuning (SFT) in generalization ability. To the best of our
knowledge, RAIDX is the first to apply the GRPO reinforcement
learning training strategy to the field of deepfake detection, which
significantly reduces reliance on manual mask annotations and text
annotations—forming the foundation of our annotation-efficient
approach.
3 METHOD
In this section, we begin by outlining the overall architecture of
our proposed RAIDX framework in Section 3.1. This framework
seamlessly integrates a vision encoder, a Retrieval-Augmented Gen-
eration (RAG) module, and a partially trainable large language
model to enable multimodal deepfake detection and explanation.
Next, in Section 3.2, we elaborate on the GRPO-based reinforcement
learning approach used for tuning LoRA [ 25] adapters, enhancing
both interpretability and robustness without the need for pixel-level
annotations or manually crafted explanation labels.
As shown in Figure 2, a central innovation of RAIDX is its seam-
less integration of Retrieval-Augmented Generation (RAG) and
Group Relative Policy Optimization (GRPO), which together en-
able precise and interpretable reasoning. The RAG module bolsters
factual accuracy by incorporating example-driven knowledge into

MM ’25, October 27–31, 2025, Dublin, Ireland Li et al.
Reference information: Among 
the {k} reference images most 
similar to the current image, 
{real_count} are labeled as REAL , 
and {fake_count} are labeled as 
FAKE .
This distribution of authenticity 
among similar images may help 
assess the current image,
but please make your judgment 
primarily based on the features of 
the image itself.
ViT
Query ImageRAG Training Images (w/labels)
Feature Vectors
FAISS Index (RAG)Retrieval
Tokenizer LLMIsthis image real?
Please carefully 
analyze this image and 
provide a detailed 
explanation.
User Prompt
Figure 3: Retrieval-Augmented Generation (RAG) module.
the prompt, while GRPO refines a lightweight LoRA explanation
head to enhance reasoning consistency and detection accuracy.
This training approach empowers the model to self-improve via
structured feedback and naturally develop Chain-of-Thought style
reasoning capabilities.
3.1 Architecture Design
We introduce RAIDX, a Retrieval-Augmented framework engi-
neered for multimodal deepfake detection and explainability. RAIDX
integrates four core components: a Vision Transformer (ViT) for
image feature extraction, a frozen tokenizer for encoding textual
instructions, a Retrieval-Augmented Generation (RAG) module for
knowledge grounding via exemplar-based reasoning, and a par-
tially trainable Large Language Model (LLM) augmented with LoRA
adapters, optimized through GRPO reinforcement learning. The
complete framework is illustrated in Figure 2. Each module is de-
scribed in detail below:
Vision Transformer. The Vision Transformer (ViT) serves as
the visual encoder to extract high-level visual features from input
images. These extracted features are used in two parallel pathways:
one is fed into the RAG module for FAISS-based retrieval, and the
other is fused with textual tokens and processed jointly by the
language model.
Tokenizer. The text tokenizer processes input prompts by convert-
ing them into a structured format. It combines user prompts with
the RAG module’s output to form a cohesive prompt sequence. An
example of a user input prompt is shown in Figure 2.Retrieval-Augmented Generation Module. To leverage prior
knowledge from similar examples, we incorporate a Retrieval-
Augmented Generation (RAG) module as a preprocessing step be-
fore inputs reach the LLM, as discussed in Section 1. Specifically,
we construct a vector index comprising all training images us-
ing FAISS [ 18], an efficient library for fast similarity search over
high-dimensional embeddings. Each image is encoded into a fixed-
dimensional vector representation using a ViT encoder. We elabo-
rate the steps below.
Given a query image 𝑥𝑞, its visual embedding 𝑣𝑞is computed and
used to retrieve the top- 𝑘most similar images {𝑥𝑟
1,𝑥𝑟
2,...,𝑥𝑟
𝑘}from
the FAISS index. Each retrieved image has an associated binary
label𝑦𝑟
𝑖∈{REAL,FAKE}. We aggregate the label distribution to
form a retrieval-based statistical summary as:
"Among the 𝑘retrieved images, 𝑛𝑟are REAL and 𝑛𝑓are FAKE. "
This summary is integrated with a fixed task prompt to create
the complete input for the LLM. The final reference information is
formatted as depicted in Figure 3. RAG enables the LLM to reason
contextually while minimizing over-reliance on retrieval bias. As
the RAG index is constructed once from training data, it incurs no
additional supervision or labeling costs.
Large Language Model. The LLM, with its core parameters frozen
and only LoRA adapters updated during training, performs dual-
stage reasoning as follows: First, it generates a <think> block con-
taining detailed reasoning steps, analyzing cues such as lighting
consistency, shadow sharpness, edge details, or semantic irregular-
ities. Then, it outputs a final <answer> block indicating whether
the image is REAL orFAKE .
Furthermore, we leverage the LLM to produce saliency maps
as visual explanations, highlighting regions most critical to the
model’s final decision. These maps are derived from attention scores
across the ViT’s layers. Specifically, we use the attention rollout
method, recursively aggregating attention weights from lower to
higher layers. The attention maps between the [CLS] token and
patch tokens are fused to generate a coarse-to-fine importance map.
Given a sequence of attention matrices {𝐴(𝑙)}𝐿
𝑙=1, the cumulative
attention is computed as:
˜𝐴=𝐴(1)𝐴(2)···𝐴(𝐿)(1)
The saliency score for each image patch, derived from the at-
tention weight between the [CLS] token and the corresponding
patch token, is upsampled to match the original image’s resolution
and visualized as a jet colormap overlay, requiring no pixel-level
annotations while producing human-interpretable heatmaps that
correlate with textual reasoning and highlight forgery artifacts
like inconsistent edges, blurred backgrounds, or distorted facial
features.
3.2 Training
To improve the factual grounding and interpretability of RAIDX,
we employ Group Relative Policy Optimization (GRPO) to fine-tune
both the ViT-based vision encoder and the LoRA adapters within the
LLM. In contrast, the base language model and the retrieval module
remain frozen throughout training to preserve pre-trained linguis-
tic and retrieval capabilities. This selective optimization ensures
that the model can adapt its visual understanding and explanatory

RAIDX: A RAG-GRPO Framework for Explainable Deepfake Detection MM ’25, October 27–31, 2025, Dublin, Ireland
reasoning while avoiding overfitting to the downstream task. Below,
we elaborate on the details of GRPO:
Reward Definition. For each query 𝑞, the model produces a
group of outputs{𝑜1,𝑜2,...,𝑜𝐺}sampled from the previous policy
𝜋𝜃old. Each output is assigned a reward 𝑟𝑖based on two criteria:
•Accuracy:𝑟acc=1if the model predicts correctly, otherwise
0.
•Format:𝑟fmt=1if the output adheres to the required struc-
ture (e.g., using <think> and<answer> blocks), otherwise
0.
The total reward is computed as:
𝑟𝑖=𝑟acc,𝑖+𝑟fmt,𝑖 (2)
Advantage Computation. To reduce variance and stabilize
optimization, we normalize the reward 𝑟𝑖using the mean 𝜇𝑟and
standard deviation 𝜎𝑟of the group rewards {𝑟1,...,𝑟𝐺}:
𝐴𝑖=𝑟𝑖−𝜇𝑟
𝜎𝑟,where𝜇𝑟=1
𝐺𝐺∑︁
𝑗=1𝑟𝑗 (3)
GRPO Objective. The optimization objective of GRPO is defined
as:
JGRPO(𝜃)=E𝑞∼𝑃(𝑄)"
1
𝐺𝐺∑︁
𝑖=1 
min𝜋𝜃(𝑜𝑖|𝑞)
𝜋𝜃old(𝑜𝑖|𝑞),
clip𝜋𝜃(𝑜𝑖|𝑞)
𝜋𝜃old(𝑜𝑖|𝑞),1−𝜖,1+𝜖
𝐴𝑖
−𝛽DKL(𝜋𝜃∥𝜋ref)!#
(4)
where𝜖and𝛽are hyper-parameters, and 𝐴𝑖is the advantage, com-
puted using a group of rewards {𝑟1,𝑟2,...,𝑟𝐺}corresponding to
the outputs within each group.
The KL-divergence penalty is computed as:
DKL(𝜋𝜃||𝜋ref)=𝜋ref(𝑜𝑖|𝑞)
𝜋𝜃(𝑜𝑖|𝑞)−log𝜋ref(𝑜𝑖|𝑞)
𝜋𝜃(𝑜𝑖|𝑞)−1
Here,𝜖and𝛽are hyperparameters, and 𝜋refrefers to a reference
policy, such as a pre-trained model or a snapshot from a previous
iteration.
4 EXPERIMENTS
In this section, we conduct a comprehensive evaluation of RAIDX,
addressing six critical research objectives: (1) systematically val-
idating detection accuracy against standardized benchmarks, (2)
assessing cross-model generalization capabilities across unseen
generative models, (3) conducting both quantitative and qualitative
evaluations on the explanation quality of RAIDX, (4) investigating
operational robustness under common real-world perturbations, (5)
conducting comprehensive ablation analyses to quantify the impact
of RAIDX’s core design choices, and (6) presenting and analyzing
both correct and failure cases generated by the RAIDX model to
provide deeper insights into its performance.4.1 Detection Performance on SID-Set
In this experiment, we adopt the SID-Set [ 27] benchmark, which
comprises real images and synthetic images generated by a diverse
set of foundation models.
Importantly, we emphasize that only the Real and Synthetic
subsets of SID-Set are used in both training and evaluation, while
the Tampered subset is excluded. This is because our task focuses
on whole-image AI-generated detection, whereas the Tampered
subset consists of partially manipulated images that fall outside the
scope of our detection objective.
We compare RAIDX with a broad set of state-of-the-art base-
lines, including traditional frequency-domain methods, CNN-based
detectors, and recent VLM-based architectures. As metrics, we use
Accuracy and F1-score, computed separately on real and fake sam-
ples.
Table 1: Performance comparison on the SID-Set.
MethodsYear Real Fake
ACC F1 ACC F1
FreDect [20] 2020 83.7 91.1 16.8 28.8
Gram-Net [44] 2020 70.1 82.4 93.5 96.6
CnnSpott [69] 2021 79.8 88.7 39.5 56.6
Fusing [29] 2022 85.1 92.0 34.0 50.7
UnivFD [48] 2023 68.0 67.4 62.1 87.5
LGrad [65] 2023 64.8 78.6 83.5 91.0
LNP [5] 2023 71.2 83.2 91.8 95.7
AntifakePrompt [7] 2024 64.8 78.6 93.8 96.8
SIDA-13B [27] 2025 96.7 97.3 98.7 99.3
RAIDX 2025 98.5 98.9 99.4 99.5
As shown in Table 1, RAIDX consistently achieves the highest
Accuracy and F1-score across both categories, highlighting its su-
perior detection capability and balanced classification performance.
Significantly, RAIDX outperforms the strong baseline SIDA-13B[ 27],
published in CVPR 2025, on each evaluation metric, demonstrating
its effectiveness.
4.2 Generalization to Unseen Generative Models
To evaluate RAIDX’s zero-shot generalization capability, we adhere
to the benchmark protocol of AntifakePrompt [ 7]. Specifically, we
train RAIDX on its official training set, comprising real images from
COCO [ 39] and Flickr[ 79] alongside synthetic images generated
by eight models: Stable Diffusion (v1.4, v1.5, v2.0, v2.1)[ 3], DALLE
2[54], Midjourney v5[46], IF[4], and GLIDE[47].
For evaluation, we leverage 18 additional datasets from the An-
tifakePrompt benchmark, all generated by unseen diffusion models
such as SDXL[ 50], DALLE-3[ 49], DiffusionDB[ 73], and GLIDE[ 47]
Stylization. These datasets span diverse styles and domains, en-
suring a rigorous test of generalization under distribution shifts.
As shown in Table 2, RAIDX outperforms a wide range of base-
lines—including frequency-based, CNN-based, and vision-language

MM ’25, October 27–31, 2025, Dublin, Ireland Li et al.
Table 2: Accuracy comparison across models and datasets. Green-shaded rows indicate Real (non-AI-generated) datasets, while
Red-shaded rows denote AI-generated datasets.
Dataset Ricker2022 [55] FatFormer [41] Wang2020 [70] DIRE [72] LASTED [76] QAD [33] ResNet [24] DE-FAKE [58] CogVLM [71] InstructBLIP [12] AntifakePrompt [7] RAIDX
COCO 95.60 97.40 96.87 81.77 75.47 59.57 99.43 85.97 98.43 97.63 92.53 92.84
Flickr 95.80 98.13 96.67 77.53 65.58 60.23 99.23 84.38 99.63 97.50 91.57 94.27
SD2 81.10 16.83 5.23 30.47 52.53 51.00 2.50 88.07 52.47 89.57 98.33 98.65
SD3 88.40 21.50 8.60 98.53 79.51 46.53 99.83 95.17 2.10 97.60 96.17 98.13
SDXL 81.10 30.39 1.53 19.73 77.65 41.60 0.50 72.17 32.57 96.47 99.17 98.77
IF 92.65 27.73 4.93 63.17 59.89 30.97 4.40 95.20 29.03 87.90 97.10 98.46
DALLE-2 52.10 76.03 3.40 61.72 59.63 15.17 12.80 61.17 60.70 99.27 97.27 97.91
DALLE-3 95.20 43.97 8.17 36.27 51.83 9.83 2.10 71.57 6.03 67.87 80.80 87.85
playground v2.5 94.40 29.83 15.73 17.73 65.42 38.73 0.20 76.07 13.37 95.43 97.73 96.94
DiffusionDB 81.20 60.50 9.64 16.40 86.48 52.07 4.69 78.10 6.05 85.40 98.47 97.58
SGXL 100.00 97.73 2.13 9.50 64.39 46.40 1.63 90.20 60.40 91.20 99.03 98.74
GLIDE 83.80 79.80 5.87 45.44 66.19 53.63 49.97 50.20 59.90 92.63 98.90 98.93
Stylization 75.50 85.03 11.40 50.76 67.79 51.93 0.90 55.17 42.90 82.80 94.10 95.28
DF 14.20 5.10 0.30 3.77 58.36 97.43 34.20 77.17 13.80 67.43 95.03 97.93
DFDC 46.90 1.60 0.00 60.13 70.12 90.40 14.20 48.57 9.00 85.47 99.83 98.84
LaMa 64.30 67.03 7.53 13.97 60.53 42.73 1.87 23.00 5.20 42.73 39.40 65.98
SD2IP 59.10 85.07 7.23 86.40 99.56 96.30 99.76 75.57 35.50 91.13 89.87 92.43
SD2SR 73.90 84.03 1.40 27.20 59.99 47.50 97.79 96.67 55.06 99.90 99.43 92.31
Average 76.40 55.98 16.23 39.48 67.83 51.78 34.78 73.58 37.90 87.10 92.48 94.55
model (VLM)-based methods such as InstructBLIP and Antifake-
Prompt—achieving the highest average accuracy across all datasets.
In particular, RAIDX demonstrates strong robustness on challeng-
ing generative sources such as GLIDE[ 47], DALLE-3[ 49], and Styl-
ization.
It is noteworthy that although models like ResNet perform well
on certain real datasets like COCO[ 39], their performance collapses
under domain shifts, highlighting the importance of robust reason-
ing and contextual retrieval. The superior performance of RAIDX
can be attributed to its integration of RAG-based contextual aug-
mentation and LoRA-enhanced policy reasoning.
4.3 Explanation Study
GRPO Study: To assess the impact of GRPO on explanation quality,
we randomly sampled 100 synthetic images from the SID-Set [27].
Each image was processed by two variants of the same base model
architecture (Qwen2.5-VL): (1) SFT, trained via supervised fine-
tuning, and (2) RAIDX , further optimized with GRPO reinforce-
ment learning. For each image, the models generated an explana-
tion for their prediction. To ensure objective and rigorous evalua-
tion, each output was independently scored by ten domain experts
blinded to model identities.
Table 3: Evaluation on the explainability.
Method Avg Score 95% CI
SFT 22.5 (20.15, 24.85)
RAIDX 82.50 (79.85, 85.15)
As shown in Table 3, RAIDX significantly outperforms SFT in
terms of explanation quality, achieving a much higher average score
with narrower confidence intervals. These results demonstrate the
effectiveness of GRPO in enhancing both the accuracy and coher-
ence of model-generated explanations, surpassing the capabilities
of supervised fine-tuning alone.
Comparison with Coarse-grained Methods: To further vali-
date the fine-grained reasoning capability of RAIDX, we compare
it with SIDA-13B [27], a state-of-the-art detector that generates
region-aware but coarse textual explanations. While SIDA providesgeneral summaries of visual anomalies, RAIDX produces detailed,
evidence-grounded justifications that align with specific visual
artifacts, demonstrating superior interpretability in high-fidelity
forensics.
Table 4: Explainability Comparison: RAIDX vs. SIDA-13B.
Method Avg Score 95% CI
SIDA-13B (Coarse-grained) 67.15 (64.89, 69.41)
RAIDX (Fine-grained) 82.50 (79.85, 85.15)
SIDA: This image is: Fully Synthetic Image 
T echnical Analysis:
1.1 Lighting and Color: The lighting appears globally warm and cinematic, but the direction and shadow 
softness are inconsistent with natural sunlight diffusion in narrow alleys.
Color Saturation: Colors are richly saturated, especially the buildings and clothing, suggesting stylization 
beyond typical photographic range...
Human Perception Indicators:
2.1 Natural Imperfections: Surfaces —especially the alley floor and walls —lack scratches, wear, or dirt 
variation expected in lived -in urban environments.
2.2 Environmental Integration: The boy and the ball appear slightly lifted from the surface; subtle cues like 
foot pressure, motion blur, or dust displacement are missing.
Is this image real?
Please carefully analyze this image and provide 
a detailed explanation.Ground truth: Fake
RAIDX: <think>
The image depicts a young boy running down a narrow cobblestone alleyway, chasing a white soccer 
ball. The lighting suggests a late afternoon or early evening setting, casting a warm golden glow across the 
scene. 
Analysis: Lighting and Shadows : While the warm directional lighting is generally consistent, the 
shadows beneath the boy  and the ball appear unnaturally sharp and lack proper blending into the 
cobblestone surface . Anatomical Irregularities: The boy’s running posture looks rigid , with the lifted foot 
lacking natural flex or weight distribution. Surface Texture and Artifacts: The cobblestone ground is overly 
smooth and uniform, lacking natural imperfections like cracks, uneven wear, or dirt . The reflections and 
highlights on the stones appear too regular, reducing material realism…
</think> <answer>FAKE</answer>
Figure 4: Comparison of explanations from RAIDX and SIDA.
Table 4 shows that RAIDX significantly outperforms SIDA-13B[ 27]
in terms of expert-rated explanation quality. Furthermore, Figure 4
illustrates a side-by-side visual comparison, where RAIDX high-
lights precise issues such as eye asymmetry and shadow misalign-
ment, in contrast to SIDA’s broader region-level masks and generic
textual feedback.

RAIDX: A RAG-GRPO Framework for Explainable Deepfake Detection MM ’25, October 27–31, 2025, Dublin, Ireland
4.4 Robustness to Visual Perturbations
Table 5: Robustness evaluation of RAIDX on SID-Set[ 27] and
multiple unseen generative datasets.
MethodsSID-Set (Test1) Average External Sets (Test2)
ACC F1 Avg-ACC Avg-F1
JPEG 90 94.21 94.02 92.35 91.97
JPEG 80 91.88 92.10 89.23 88.67
Gaussian 10 95.72 95.41 93.68 93.21
Resize 224 97.33 97.21 94.83 94.65
Bright+Gaussian 93.45 93.16 91.28 90.84
RAIDX 98.95 99.20 94.55 94.92
For robustness evaluation, we apply five commonly observed distor-
tions: JPEG compression (quality 90 and 80), Gaussian blur ( 𝜎=10),
resizing to 224×224, and brightness reduction with added Gaussian
noise. We report performance on two settings: (1) Test1, which ap-
plies these perturbations to the SID-Set[ 27] test split, and (2) Test2,
which applies the same perturbations to 10 unseen datasets from
the AntifakePrompt test split—namely SD2, SD3, SDXL[ 50], IF[ 4],
DALLE-2[ 54], DALLE-3[ 49], playground v2.5[ 36], DiffusionDB[ 73],
and GLIDE[ 47]. As shown in Table 5, RAIDX was not explicitly
trained on distorted images, but it consistently maintains high
performance across all degradation types. The accuracy drop com-
pared to clean images is minimal on both test sets, indicating strong
generalization capability. These results highlight the model’s ro-
bustness to real-world noise and compression artifacts, reinforcing
its practical utility for deployment in noisy, user-generated content
environments.
4.5 Ablation Studies
We conduct ablation studies to analyze the individual contributions
of the RAG and GRPO modules.
RAG Ablation: To assess the impact of retrieval-based guidance
in RAIDX, we compare three model variants:
1) No RAG: The model processes only the raw input image with
a generic instruction prompt, without any retrieved context.
2) Static Prompt: Instead of dynamic retrieval, a fixed prompt
summarizes neighborhood prior statistics in the form: “Reference
information: Among the {k} reference images most similar to the
current image, {real_count} are labeled as REAL, and {fake_count} are
labeled as FAKE. ”
3) RAIDX: Our full model retrieves top- 𝑘similar images using
a FAISS index over ViT embeddings. The class distribution of re-
trieved samples is integrated as contextual guidance for prompt
generation.
As shown in Table 6, RAIDX outperforms both baselines, im-
proving average accuracy by 1.45% over “No RAG” and 1.48% over
“Static Prompt”. The gains are especially notable on challenging
datasets (e.g., DFDC[ 16], GLIDE[ 47]), confirming the importance
of retrieval-grounded instruction over generic or pseudo-context
prompts.Table 6: Ablation study of RAG module on detection perfor-
mance (Accuracy %).
Dataset No RAG Static Prompt RAIDX
SD2 97.42 97.56 97.65
DALLE-2 94.88 94.13 97.91
DALLE-3 87.11 87.27 87.85
LaMa 64.83 65.48 65.98
GLIDE 97.37 97.01 98.31
DFDC 96.28 96.26 98.84
Average 89.64 89.61 91.09
GRPO Ablation:
To rigorously quantify GRPO’s direct impact on classification
performance, we performed an ablation on the SID-Set benchmark,
holding all data splits, model architecture, and hyperparameters
constant. Table 7 presents the results:
Table 7: Ablation of GRPO’s impact on detection accuracy
Model Variant Accuracy vs Base
Base 57.12% —
Base + GRPO 87.45% +30.33
RAIDX (Base + RAG + GRPO) 93.07% +35.95
Table 7 shows that adding GRPO to the base model yields a large
+30.33% gain in accuracy, which further improves by +5.62% when
combined with RAG. This step-wise improvement shows that GRPO
contributes a substantial and independent gain by jointly optimiz-
ing classification accuracy and explanation quality. These results
validate GRPO as a critical component of RAIDX’s effectiveness.
4.6 Case Studies
In this section, we present a set of representative examples from
the SID-Set[ 27], visualized in Figure 5. We categorize them into
three types: (a) Correct Examples, correctly classifies AI-generated
images and produces detailed, visually grounded explanations, (b)
Failure Saliency Map Examples, still makes correct predictions, but
the saliency heatmaps fail to accurately localize informative regions,
and (c) Failure Explanation Examples, illustrates cases where the
prediction is incorrect and the explanation lacks visual or semantic
specificity.
Overall, RAIDX demonstrates robust performance in both pre-
diction and explanation across most examples. However, the case
studies reveal two major failure modes: saliency map drift and
failure explanation. These findings reveal two main areas for im-
provement: improving visual grounding accuracy and encouraging
explanations that focus more explicitly on AI-generated artifacts.
5 CONCLUSION AND DISCUSSION
In this work, we introduce RAIDX, a novel LLM-based deepfake de-
tection framework that integrates Retrieval-Augmented Generation

MM ’25, October 27–31, 2025, Dublin, Ireland Li et al.
<think>
The image depicts 
two children playing 
on a beach. They 
appear …The  facial 
features lack fine -
grained
detail…Both 
children exhibit 
fingers that are 
either fused or 
rendered with 
insufficient 
anatomical 
separation …Childre
n’s feet show 
unnatural blending 
into the sand …The 
ball’s texture lacks 
micro -level…
</think>
<answer>FAKE</an
swer>
(b) Failure Saliency Map Examples
(a) Correct ExamplesIs this image real?
Please carefully analyze this image and provide 
a detailed explanation.
<think>
The image depicts a 
child  standing in a 
snowy 
landscape …The  
facial features of 
the child are 
unnaturally smooth 
lacking the micro -
texture …The 
gloves appear 
slightly melted into 
the jacket cuffs, 
with imprecise 
edges and 
inconsistent …The 
knitted texture of 
the hat also lacks 
depth and fine fiber 
detail…
</think>
<answer>FAKE</an
swer><think>
The image shows a 
skier  mid -air against 
a snowy landscape 
backdrop… The 
skier’s hands and 
gloves  exhibit 
unnatural stiffness 
and blending with 
the ski poles , 
lacking joint 
articulation and fine 
detail…The  skis
show inconsistent 
shading and an 
overly smooth …The 
snow -covered trees 
in the background  
lack of depth 
variation …
</think>
<answer>FAKE</an
swer><think>
The image depicts a 
black -and -white 
dog  swimming 
directly toward the 
camera…The 
symmetry of the 
ears and the glossy 
texture of the eyes 
appear unusually 
uniform and 
artistically 
enhanced …The 
nose and snout 
region  are too 
smooth and 
rounded , without 
visible wetness or 
texture 
variation …</think>
<answer>FAKE</an
swer><think>
The image shows 
a young child 
playing outdoors 
in a sunlit grassy 
field…The facial 
features lack 
fine -grained
detail…
</think>
<answer>REAL</
answer>
<think>
The image shows 
a young boy on a 
city street during  
golden hour… 
exhibit realistic 
reflections and  
soft lens …
</think>
<answer>REAL</
answer><think>
The image 
depicts a close -
up view of 
grilled food on 
a barbecue, 
with a person’s 
hand lifting …
</think>
<answer>REAL
</answer><think>
The image 
captures a 
candid and 
serene moment 
between a 
woman and a 
small child sitting 
near a lake …
</think>
<answer>REAL</
answer>
(c) Failure Explanation Examples
Figure 5: Visualization results from RAIDX, including both correct cases and failure cases.
(RAG) and Group Relative Policy Optimization (GRPO) to simulta-
neously enhance detection accuracy and decision explainability. By
unifying retrieval-augmented exemplar context with reinforcement
learning for Chain-of-Thought reasoning, RAIDX combines detec-
tion and explanation in a single pipeline. The proposed architecture
generates textual rationales and attention-based heatmaps, achiev-
ing dual-modality explainability. Extensive experiments on SID-
Set[27] and 18 unseen generative datasets demonstrate RAIDX’s
superiority over state-of-the-art methods in accuracy, generaliza-
tion, and robustness under perturbations. Ablation studies further
validate the critical roles of RAG retrieval, GRPO training, and
reward design strategies in RAIDX’s performance.
Limitations: As the first paper in the deepfake detection field
to employ RAG (Retrieval-Augmented Generation) and GRPO rein-
forcement learning training strategies, RAIDX has demonstrated
promising results. However, as an initial attempt, it still exhibits
the following limitations: 1. Optimization of RAG Implementa-
tion: Current strategies for utilizing RAG leave substantial room
for exploration. A critical challenge lies in efficiently integrating
unseen data into external knowledge with minimal cost (e.g., gener-
ating only 5-10 corresponding synthetic samples per model). Thiswould enable rapid adaptation to future unseen generative mod-
els without requiring adjustments to existing model parameters.
2. Limitations in Tampered Image Detection: While RAIDX
has achieved notable performance in visualizing saliency maps of
potential fake regions in fully synthetic images, it currently fails
to address cases of image tampering. Enhancing its capability to
handle both synthetic content and manipulated authentic media
will constitute a key focus of our future research to expand its
applicability across diverse scenarios.
Future directions: In addition to the future work outlined in the
Limitations part, our planned research will further extend RAIDX to
video-based deepfake detection through temporal modeling and de-
velop human-aligned evaluation metrics for assessing explanation
quality using human feedback or LLMs.

RAIDX: A RAG-GRPO Framework for Explainable Deepfake Detection MM ’25, October 27–31, 2025, Dublin, Ireland
ACKNOWLEDGMENTS
This work is supported by The Alan Turing Institute (UK) through
the project Turing-DSO Labs Singapore Collaboration (SDCfP2\100009).
REFERENCES
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al. 2023. Gpt-4 technical report. Arxiv (2023).
[2]Darius Afchar, Vincent Nozick, Junichi Yamagishi, and Isao Echizen. 2018.
Mesonet: a compact facial video forgery detection network. In WIFS .
[3]Stability AI. 2022. Stable Diffusion v2.0. https://huggingface.co/stabilityai/stable-
diffusion-2. Accessed: 2025-04-12.
[4]Stability AI. 2023. DeepFloyd IF. https://github.com/deep-floyd/IF. Accessed:
2025-04-12.
[5]Xiuli Bi, Bo Liu, Fan Yang, Bin Xiao, Weisheng Li, Gao Huang, and Pamela C.
Cosman. 2023. Detecting Generated Images by Real Images Only. Arxiv (2023).
[6]Junyi Cao, Chao Ma, Taiping Yao, Shen Chen, Shouhong Ding, and Xiaokang
Yang. 2022. End-to-End Reconstruction-Classification Learning for Face Forgery
Detection. In CVPR .
[7]You-Ming Chang, Chen Yeh, Wei-Chen Chiu, and Ning Yu. 2023. AntifakePrompt:
Prompt-Tuned Vision-Language Models are Fake Image Detectors. Arxiv (2023).
[8]Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. 2025. R1-V: Reinforcing
Super Generalization Ability in Vision-Language Models with Less Than $3.
https://github.com/Deep-Agent/R1-V. Accessed: 2025-02-02.
[9]Liang Chen, Yong Zhang, Yibing Song, Lingqiao Liu, and Jue Wang. 2022. Self-
supervised Learning of Adversarial Example: Towards Good Generalizations for
Deepfake Detection. Arxiv (2022).
[10] François Chollet. 2017. Xception: Deep Learning with Depthwise Separable
Convolutions. Arxiv (2017).
[11] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, and Mubarak Shah.
2023. Diffusion Models in Vision: A Survey. PAMI (2023).
[12] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao,
Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. 2023. InstructBLIP:
Towards General-purpose Vision-Language Models with Instruction Tuning.
arXiv:2305.06500 [cs.CV] https://arxiv.org/abs/2305.06500
[13] Alan Dao and Dinh Bach Vu. 2025. AlphaMaze: Enhancing Large Language
Models’ Spatial Intelligence via GRPO. Arxiv (2025).
[14] DeepSeek-AI. 2024. DeepSeek-V3 Technical Report. Arxiv (2024).
[15] DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
via Reinforcement Learning. Arxiv (2025).
[16] Brian Dolhansky, Joanna Bitton, Ben Pflaum, Jikuo Lu, Russ Howes, Menglin
Wang, and Cristian Canton Ferrer. 2020. The DeepFake Detection Challenge
(DFDC) Dataset. Arxiv (2020).
[17] Chengbo Dong, Xinru Chen, Ruohan Hu, Juan Cao, and Xirong Li. 2023. MVSS-
Net: Multi-View Multi-Scale Supervised Networks for Image Manipulation De-
tection. T-PAMI (2023).
[18] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2025.
The Faiss library. Arxiv (2025).
[19] Junxian Duan, Yuang Ai, Jipeng Liu, Shenyuan Huang, Huaibo Huang, Jie Cao,
and Ran He. 2024. Test-time Forgery Detection with Spatial-Frequency Prompt
Learning. ICCV (2024).
[20] Joel Frank, Thorsten Eisenhofer, Lea Schönherr, Asja Fischer, Dorothea Kolossa,
and Thorsten Holz. 2020. Leveraging frequency analysis for deep fake image
recognition. In ICML . 3247–3258.
[21] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Ji-
awei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Generation
for Large Language Models: A Survey. Arxiv (2024).
[22] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-
Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative
Adversarial Networks. Arxiv (2014).
[23] Xiao Guo, Xiaohong Liu, Zhiyuan Ren, Steven Grosz, Iacopo Masi, and Xiaoming
Liu. 2023. Hierarchical Fine-Grained Image Forgery Detection and Localization.
InCVPR .
[24] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. Deep Residual
Learning for Image Recognition. Arxiv (2015).
[25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large
Language Models. In ICLR .
[26] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao,
Zhe Xu, Yao Hu, and Shaohui Lin. 2025. Vision-R1: Incentivizing Reasoning
Capability in Multimodal Large Language Models. Arxiv (2025).
[27] Zhenglin Huang, Jinwei Hu, Xiangtai Li, Yiwei He, Xingyu Zhao, Bei Peng,
Baoyuan Wu, Xiaowei Huang, and Guangliang Cheng. 2025. SIDA: Social Media
Image Deepfake Detection, Localization and Explanation with Large Multimodal
Model. In CVPR .[28] Zhenglin Huang, Tianxiao Li, Xiangtai Li, Haiquan Wen, Yiwei He, Jiangning
Zhang, Hao Fei, Xi Yang, Xiaowei Huang, Bei Peng, and Guangliang Cheng. 2025.
So-Fake: Benchmarking and Explaining Social Media Image Forgery Detection.
Arxiv (2025).
[29] Yan Ju, Shan Jia, Lipeng Ke, Hongfei Xue, Koki Nagano, and Siwei Lyu. 2022.
Fusing Global and Local Features for Generalized AI-Synthesized Image Detection.
InICIP.
[30] Hengrui Kang, Siwei Wen, Zichen Wen, Junyan Ye, Weijia Li, Peilin Feng,
Baichuan Zhou, Bin Wang, Dahua Lin, Linfeng Zhang, and Conghui He. 2025.
LEGION: Learning to Ground and Explain for Synthetic Image Detection. Arxiv
(2025).
[31] Tero Karras, Samuli Laine, and Timo Aila. 2021. A Style-Based Generator Archi-
tecture for Generative Adversarial Networks. PAMI (2021).
[32] Rohit Kundu, Hao Xiong, Vishal Mohanty, Athula Balachandran, and Amit K
Roy-Chowdhury. 2024. Towards a Universal Synthetic Video Detector: From
Face or Background Manipulations to Fully AI-Generated Content. Arxiv (2024).
[33] Binh M. Le and Simon S. Woo. 2023. Quality-Agnostic Deepfake Detection with
Intra-model Collaborative Learning. Arxiv (2023).
[34] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. Arxiv (2021).
[35] Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Genera-
tion for Knowledge-Intensive NLP Tasks. In NeurIPS .
[36] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi.
2024. Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in
Text-to-Image Generation. Arxiv (2024).
[37] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping
language-image pre-training for unified vision-language understanding and
generation. In ICML .
[38] Jingchun Lian, Lingyu Liu, Yaxiong Wang, Yujiao Wu, Li Zhu, and Zhedong
Zheng. 2024. A Large-scale Interpretable Multi-modality Benchmark for Facial
Image Forgery Localization. Arxiv (2024).
[39] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick,
James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár.
2015. Microsoft COCO: Common Objects in Context. Arxiv (2015).
[40] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruc-
tion tuning. NeurIPS (2023).
[41] Huan Liu, Zichang Tan, Chuangchuang Tan, Yunchao Wei, Yao Zhao, and Jing-
dong Wang. 2023. Forgery-aware Adaptive Transformer for Generalizable Syn-
thetic Image Detection. Arxiv (2023).
[42] Jiawei Liu, Fanrui Zhang, Jiaying Zhu, Esther Sun, Qiang Zhang, and Zheng-Jun
Zha. 2025. ForgeryGPT: Multimodal Large Language Model For Explainable
Image Forgery Detection and Localization. Arxiv (2025).
[43] Xiaohong Liu, Yaojie Liu, Jun Chen, and Xiaoming Liu. 2022. PSCC-Net: Progres-
sive Spatio-Channel Correlation Network for Image Manipulation Detection and
Localization. T-CSVT (2022).
[44] Zhengzhe Liu, Xiaojuan Qi, and Philip H. S. Torr. 2020. Global Texture Enhance-
ment for Fake Face Detection in the Wild. In CVPR .
[45] Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang, Jiayi
Ji, Fei Chao, Jiebo Luo, and Rongrong Ji. 2024. Video-RAG: Visually-aligned
Retrieval-Augmented Long Video Comprehension. ArXiv (2024).
[46] Inc. Midjourney. 2023. Midjourney v5. https://www.midjourney.com/. Accessed:
2025-04-12.
[47] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin,
Bob McGrew, Ilya Sutskever, and Mark Chen. 2022. GLIDE: Towards Photoreal-
istic Image Generation and Editing with Text-Guided Diffusion Models. Arxiv
(2022).
[48] Utkarsh Ojha, Yuheng Li, and Yong Jae Lee. 2023. Towards Universal Fake Image
Detectors that Generalize Across Generative Models. In CVPR .
[49] OpenAI. 2023. DALL ·E 3. https://openai.com/dall-e. Accessed: 2025-04-12.
[50] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn,
Jonas Müller, Joe Penna, and Robin Rombach. 2023. SDXL: Improving Latent
Diffusion Models for High-Resolution Image Synthesis. Arxiv (2023).
[51] Jingyuan Qi, Zhiyang Xu, Rulin Shao, Yang Chen, Di Jin, Yu Cheng, Qifan Wang,
and Lifu Huang. 2024. RoRA-VLM: Robust Retrieval-Augmented Vision Language
Models. ArXiv (2024).
[52] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al.2021. Learning transferable visual models from natural language supervision.
InICML .
[53] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al .2018.
Improving language understanding by generative pre-training.
[54] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen.
2022. Hierarchical Text-Conditional Image Generation with CLIP Latents. Arxiv
(2022).

MM ’25, October 27–31, 2025, Dublin, Ireland Li et al.
[55] Jonas Ricker, Simon Damm, Thorsten Holz, and Asja Fischer. 2024. Towards the
Detection of Diffusion Model Deepfakes. Arxiv (2024).
[56] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn
Ommer. 2022. High-Resolution Image Synthesis with Latent Diffusion Models.
InCVPR .
[57] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
2017. Proximal policy optimization algorithms. Arxiv (2017).
[58] Zeyang Sha, Zheng Li, Ning Yu, and Yang Zhang. 2023. DE-FAKE: Detection
and Attribution of Fake Images Generated by Text-to-Image Generation Models.
Arxiv (2023).
[59] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan
Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the Limits
of Mathematical Reasoning in Open Language Models. Arxiv (2024).
[60] Haozhan Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, and
Tiancheng Zhao. 2025. VLM-R1: A stable and generalizable R1-style Large Vision-
Language Model. https://github.com/om-ai-lab/VLM-R1. Accessed: 2025-02-15.
[61] Ensheng Shi, Yanlin Wang, Wei Tao, Lun Du, Hongyu Zhang, Shi Han, Dongmei
Zhang, and Hongbin Sun. 2022. RACE: Retrieval-Augmented Commit Message
Generation. Arxiv (2022).
[62] Xincheng Shuai, Henghui Ding, Xingjun Ma, Rongcheng Tu, Yu-Gang Jiang,
and Dacheng Tao. 2024. A Survey of Multimodal-Guided Image Editing with
Text-to-Image Diffusion Models. Arxiv (2024).
[63] Zhihao Sun, Haoran Jiang, Haoran Chen, Yixin Cao, Xipeng Qiu, Zuxuan Wu, and
Yu-Gang Jiang. 2024. ForgerySleuth: Empowering Multimodal Large Language
Models for Image Manipulation Detection. Arxiv (2024).
[64] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, and Yunchao
Wei. 2024. Rethinking the up-sampling operations in cnn-based generative
network for generalizable deepfake detection. In CVPR .
[65] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, and Yunchao Wei. 2023.
Learning on gradients: Generalized artifacts representation for gan-generated
images detection. In CVPR .
[66] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lam-
ple. 2023. LLaMA: Open and Efficient Foundation Language Models. Arxiv
(2023).
[67] Chengrui Wang and Weihong Deng. 2021. Representative Forgery Mining for
Fake Face Detection. Arxiv (2021).
[68] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin
Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du,
Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang
Lin. 2024. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the
World at Any Resolution. Arxiv (2024).
[69] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A
Efros. 2020. CNN-generated images are surprisingly easy to spot... for now. In
CVPR .
[70] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A.
Efros. 2020. CNN-generated images are surprisingly easy to spot... for now.
arXiv:1912.11035 [cs.CV] https://arxiv.org/abs/1912.11035
[71] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui
Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao
Dong, Ming Ding, and Jie Tang. 2024. CogVLM: Visual Expert for Pretrained
Language Models. Arxiv (2024).
[72] Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong
Chen, and Houqiang Li. 2023. DIRE for Diffusion-Generated Image Detection.
Arxiv (2023).
[73] Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover,
and Duen Horng Chau. 2023. DiffusionDB: A Large-scale Prompt Gallery Dataset
for Text-to-Image Generative Models. Arxiv (2023).
[74] Haiquan Wen, Yiwei He, Zhenglin Huang, Tianxiao Li, Zihan Yu, Xingru Huang,
Lu Qi, Baoyuan Wu, Xiangtai Li, and Guangliang Cheng. 2025. BusterX: MLLM-
Powered AI-Generated Video Forgery Detection and Explanation. Arxiv (2025).
[75] Haiquan Wen, Tianxiao Li, Zhenglin Huang, Yiwei He, and Guangliang Cheng.
2025. BusterX++: Towards Unified Cross-Modal AI-Generated Content Detection
and Explanation with MLLM. Arxiv (2025).
[76] Haiwei Wu, Jiantao Zhou, and Shile Zhang. 2023. Generalizable Synthetic Image
Detection via Language-guided Contrastive Learning. Arxiv (2023).
[77] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen,
Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and
Junyang Lin. 2025. Qwen2.5-Omni Technical Report. Arxiv (2025).
[78] Zhiyuan Yan, Taiping Yao, Shen Chen, Yandan Zhao, Xinghe Fu, Junwei Zhu,
Donghao Luo, Chengjie Wang, Shouhong Ding, Yunsheng Wu, and Li Yuan. 2024.
DF40: Toward Next-Generation Deepfake Detection. In NeurIPS .
[79] Peter Young, Alice Lai, Micah Hodosh, and J. Hockenmaier. 2014. From image
descriptions to visual denotations: New similarity metrics for semantic infer-
ence over event descriptions. Transactions of the Association for Computational
Linguistics 2 (2014), 67–78.[80] Fangneng Zhan, Yingchen Yu, Rongliang Wu, Jiahui Zhang, Shijian Lu, Lingjie
Liu, Adam Kortylewski, Christian Theobalt, and Eric P. Xing. 2023. Multimodal
Image Synthesis and Editing: The Generative AI Era. PAMI (2023).
[81] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao,
Jian-Guang Lou, and Weizhu Chen. 2023. RepoCoder: Repository-Level Code
Completion Through Iterative Retrieval and Generation. Arxiv (2023).
[82] Yinglin Zheng, Jianmin Bao, Dong Chen, Ming Zeng, and Fang Wen. 2021. Ex-
ploring temporal coherence for more general video face forgery detection. In
CVPR .