# RAVID: Retrieval-Augmented Visual Detection: A Knowledge-Driven Approach for AI-Generated Image Identification

**Authors**: Mamadou Keita, Wassim Hamidouche, Hessen Bougueffa Eutamene, Abdelmalik Taleb-Ahmed, Abdenour Hadid

**Published**: 2025-08-05 23:10:56

**PDF URL**: [http://arxiv.org/pdf/2508.03967v1](http://arxiv.org/pdf/2508.03967v1)

## Abstract
In this paper, we introduce RAVID, the first framework for AI-generated image
detection that leverages visual retrieval-augmented generation (RAG). While RAG
methods have shown promise in mitigating factual inaccuracies in foundation
models, they have primarily focused on text, leaving visual knowledge
underexplored. Meanwhile, existing detection methods, which struggle with
generalization and robustness, often rely on low-level artifacts and
model-specific features, limiting their adaptability. To address this, RAVID
dynamically retrieves relevant images to enhance detection. Our approach
utilizes a fine-tuned CLIP image encoder, RAVID CLIP, enhanced with
category-related prompts to improve representation learning. We further
integrate a vision-language model (VLM) to fuse retrieved images with the
query, enriching the input and improving accuracy. Given a query image, RAVID
generates an embedding using RAVID CLIP, retrieves the most relevant images
from a database, and combines these with the query image to form an enriched
input for a VLM (e.g., Qwen-VL or Openflamingo). Experiments on the
UniversalFakeDetect benchmark, which covers 19 generative models, show that
RAVID achieves state-of-the-art performance with an average accuracy of 93.85%.
RAVID also outperforms traditional methods in terms of robustness, maintaining
high accuracy even under image degradations such as Gaussian blur and JPEG
compression. Specifically, RAVID achieves an average accuracy of 80.27% under
degradation conditions, compared to 63.44% for the state-of-the-art model
C2P-CLIP, demonstrating consistent improvements in both Gaussian blur and JPEG
compression scenarios. The code will be publicly available upon acceptance.

## Full Text


<!-- PDF content starts -->

RA VID: R etrieval-A ugmented Vi sual D etection: A Knowledge-Driven Approach
for AI-Generated Image Identification
Mamadou Keita
Laboratory of IEMN, Univ.
Polytechnique Hauts-de-France
Valeneciennes, Fance
mamadou.keita@uphf.frWassim Hamidouche
KU 6G Research Center,
Khalifa University
Abu Dhabi, UAE
whamidouche@gmail.com
Hessen Bougueffa Eutamene
Laboratory of IEMN, Univ.
Polytechnique Hauts-de-France
Valeneciennes, Fance
Hessen.BougueffaEutamene@uphf.frAbdelmalik Taleb-Ahmed
Laboratory of IEMN, Univ.
Polytechnique Hauts-de-France
Valeneciennes, Fance
abdelmalik.taleb-ahmed@uphf.fr
Abdenour Hadid
Sorbonne Center for Artificial Intelligence,
Sorbonne University, Abu Dhabi, UAE
abdenour.hadid@ieee.org
Abstract
In this paper, we introduce RAVID, the first framework
for AI-generated image detection that leverages visual
retrieval-augmented generation (RAG). While RAG meth-
ods have shown promise in mitigating factual inaccuracies
in foundation models, they have primarily focused on text,
leaving visual knowledge underexplored. Meanwhile, exist-
ing detection methods, which struggle with generalization
and robustness, often rely on low-level artifacts and model-
specific features, limiting their adaptability. To address this,
RAVID dynamically retrieves relevant images to enhance
detection. Our approach utilizes a fine-tuned CLIP im-
age encoder, RAVID CLIP , enhanced with category-related
prompts to improve representation learning. We further in-
tegrate a vision-language model (VLM) to fuse retrieved
images with the query, enriching the input and improving
accuracy. Given a query image, RAVID generates an em-
bedding using RAVID CLIP , retrieves the most relevant im-
ages from a database, and combines these with the query
image to form an enriched input for a VLM (e.g., Qwen-VL
or Openflamingo). Experiments on the UniversalFakeDe-
tect benchmark, which covers 19 generative models, show
that RAVID achieves state-of-the-art performance with an
average accuracy of 93.85%. RAVID also outperforms tra-
ditional methods in terms of robustness, maintaining highaccuracy even under image degradations such as Gaussian
blur and JPEG compression. Specifically, RAVID achieves
an average accuracy of 80.27% under degradation condi-
tions, compared to 63.44% for the state-of-the-art model
C2P-CLIP , demonstrating consistent improvements in both
Gaussian blur and JPEG compression scenarios. The code
will be publicly available upon acceptance.
50 60 70 80 90 100
Accuracy (%)Freq-specCo-occurenceCNN-SpotPatchforF3NetBi-LORALGradUniFDAntiFakePFreqNetNPRFatFormerRINEC2P-CLIPRAVID
55.45%66.86%69.58%71.24%71.33%78.59%80.28%81.38%82.42%85.09%87.56%90.86%91.31%93.79%93.85%
Figure 1. Performance comparison of detectors on the Uni-
versalFakeDetect dataset, including frequency-based (Freq-spec,
Co-occurrence), convolutional (CNN-Spot, F3Net), transformer-
based (FatFormer, RINE, C2P-CLIP), hybrid (Bi-LORA, LGrad),
multimodal (AntifakePrompt, Bi-LORA) architectures and RAG
(RA VID (ours)).
1arXiv:2508.03967v1  [cs.CV]  5 Aug 2025

1. Introduction
The rapid advancement of generative models, particularly
in image synthesis, has introduced significant challenges in
distinguishing AI-generated content from real data. For in-
stance, generative adversarial networks (GANs) [8, 19, 20]
and diffusion-based models [11, 31, 37, 40] have become
increasingly proficient at producing photorealistic images
that are nearly indistinguishable from genuine ones. How-
ever, the progress in detection methods has not kept pace
with these advancements, creating an urgent need for ro-
bust and reliable detection systems. As generative models
continue to evolve, so do the complexities associated with
their detection, leading to a constant race between improv-
ing generative techniques and enhancing detection capabil-
ities.
Traditional AI-generated image detection approaches
primarily rely on identifying low-level artifacts or model-
specific fingerprints [44] present in synthetic images. These
artifacts include pixel inconsistencies, noise patterns, and
subtle distortions that reveal traces of the underlying gen-
eration process. While these methods have demonstrated
effectiveness in controlled settings, they often fail in real-
world scenarios. As generative models improve, they be-
come more adept at minimizing artifacts and replicating the
statistical properties of real images, making detection in-
creasingly difficult. Furthermore, many existing detection
methods suffer from a fundamental limitation: over-reliance
on model-specific features and low-level artifacts. Since
these methods are often tailored to exploit weaknesses in
particular architectures, they struggle to generalize across
different generative models. Consequently, there is a grow-
ing need for more adaptive detection approaches that lever-
age additional sources of information to enhance perfor-
mance, robustness, and reliability.
One promising direction is retrieval-augmented genera-
tion (RAG) [24], a paradigm initially developed to improve
factual accuracy in large language models by retrieving
and incorporating external knowledge relevant to a given
query. While extensively explored in textual tasks, its po-
tential for visual tasks, particularly AI-generated image de-
tection, remains largely underexplored. Existing detection
methods mainly rely on hand-crafted features, model finger-
prints, and deep learning-based classifiers trained on limited
datasets [9, 51]. These approaches face significant chal-
lenges in generalization, struggling to detect images gener-
ated by unseen models, and robustness, as even minor per-
turbations, such as noise injection, compression artifacts, or
adversarial attacks, can significantly degrade their perfor-
mance. Overcoming these limitations requires a more adap-
tive, retrieval-based framework that dynamically integrates
external visual knowledge to improve decision-making.
To address this gap, we propose RA VID, a novel
retrieval-augmented framework for AI-generated image de-tection. Unlike traditional methods that rely solely on
model-dependent features, RA VID retrieves visually simi-
lar images relevant to the input query and integrates them
into the detection process, thereby enhancing accuracy
and robustness. At its core, RA VID leverages a CLIP-
based image encoder, fine-tuned through category-level
prompt integration to improve its ability to capture seman-
tic and structural patterns crucial for distinguishing AI-
generated images from real ones. Additionally, we in-
corporate vision-language models (VLMs), such as Open-
flamingo [2], to effectively fuse retrieved images with
the query input, enabling richer contextual understanding.
By combining retrieval-based augmentation with advanced
vision-language decision-making ability, our approach sig-
nificantly improves adaptability and effectiveness, mak-
ing it well-suited for real-world applications where reli-
able AI-generated content detection is critical. To eval-
uate our approach, we conduct extensive experiments on
UniversalFakeDetect, a large-scale benchmark comprising
AI-generated images from 19 generative models. This di-
verse dataset allows for rigorous assessment of our frame-
work’s generalization and robustness across in- and out-of-
domain scenarios. Experimental results demonstrate that
RA VID consistently outperforms existing detection tech-
niques, achieving an average accuracy of 93.85% and ex-
hibiting better generalization across multiple challenging
settings. As depicted in Figure 1, RA VID achieves the
highest accuracy, outperforming all compared state-of-the-
art detectors. In addition, RA VID also demonstrates strong
resilience to image degradations, achieving 80.27% aver-
age accuracy under conditions such as Gaussian blur and
JPEG compression significantly outperforming the state-of-
the-art C2P-CLIP model, which achieves 63.44% under the
same settings. Unlike traditional methods, it maintains high
accuracy by leveraging retrieval-augmented generation to
compensate for lost visual features, ensuring robust perfor-
mance even under real-world distortions. This highlights
the effectiveness of retrieval-augmented techniques in en-
hancing generalization and robustness across different im-
age generators. Our key contributions are summarized as
follows:
• A novel retrieval-augmented framework for AI-generated
image detection, dynamically retrieving and integrating
external visual knowledge to enhance decision-making.
• Fine-tuning of a CLIP-based image encoder with
category-level prompt integration, improving representa-
tion learning for retrieval tasks.
• Integration of VLMs to combine retrieved images with
queries, enhancing contextual understanding and detec-
tion robustness.
• Comprehensive evaluation on UniversalFakeDetect,
demonstrating significant improvements in generaliza-
tion and robustness over existing methods.
2

The rest of this paper is structured as follows. Section 2
reviews related work on AI-generated image detection, AI
image generation, and visual RAG. Section 3 introduces
the proposed visual RAG-based approach for AI-generated
image identification. Section 4 evaluates and analyzes the
performance of the proposed detection framework. Finally,
Section 7 summarizes the findings and concludes the paper.
2. Related Works
2.1. Image Generation
Recent advances in deep generative models have signifi-
cantly enhanced the capabilities of synthetic image gen-
eration. Early breakthroughs included the introduction of
GANs by Goodfellow et al. [12], enabling the creation of
realistic images without relying on input data. Subsequent
research refined GANs to improve image quality and di-
versity, and to introduce conditional generation, leading to
models like ProGAN [19], StyleGAN [20], BigGAN [3],
CycleGAN [54], StarGAN [8], and GauGAN [34]. More
recently, diffusion models have gained prominence, partic-
ularly for text-to-image synthesis. Models like Glide [31],
latent diffusion model (LDM) [40], ablated diffusion model
(ADM) [11], DALL-E 3 [33], Midjourney v5 [28], Fire-
fly [1], Imagen [42], SDXL [35], and SGXL [43] have
demonstrated impressive quality and versatility across di-
verse categories and scenes, often surpassing GANs-based
models in flexibility and image fidelity.
2.2. Visual Retrieval-Augmented Generation
RAG enhances foundation models by integrating external
knowledge into the generation process. It comprises two
key components: retrieval and generation. Given a query
q, the retrieval module selects relevant documents Kfrom
an external knowledge source (e.g., Wikipedia) based on
similarity measures. These retrieved elements serve as ad-
ditional input to the generation module, which then pro-
duces the final response yusing a causal model such as
a large language model (LLM). Despite its effectiveness
in text-based tasks, traditional RAG models primarily fo-
cus on textual data, leaving visual knowledge largely unex-
plored. Recently, a few studies have begun exploring vi-
sual RAG [39, 52]. For instance, Riedler et al. [39] investi-
gated multimodal RAG in industrial applications, assessing
image-text integration through multimodal embeddings and
textual summaries generated by GPT-4V and LLaV A [26].
Their findings indicate that while textual summaries im-
prove performance, image retrieval remains a significant
challenge. To address these limitations, Yu et al. [52] in-
troduced VisRAG, a VLM-based RAG framework that en-
codes documents as images for retrieval. By preserving
visual and layout information, VisRAG enhances genera-tion quality and outperforms text-based RAG by 20 to 40%.
Similarly, Ren et al. [38] proposed VideoRAG, a framework
designed for long-context video understanding. Their ap-
proach integrates graph-based textual grounding and mul-
timodal context encoding, efficiently preserving both se-
mantic relationships and visual features. While exist-
ing video-integrated RAG methods often predefine query-
related videos or convert them into text, thereby losing mul-
timodal richness, Jeong et al. [15] proposed an improved
VideoRAG framework. Their approach dynamically re-
trieves relevant videos and leverages VLMs to integrate vi-
sual and textual data, enabling enhanced multimodal re-
sponse generation.
2.3. Detection Methods
Detecting AI-generated images has become increasingly
critical with the proliferation of synthetic content. As im-
age generation methods evolve, researchers have developed
various techniques to enhance detection accuracy and gen-
eralization. These approaches can be broadly categorized
into three domains: spatial, frequency, and multimodal fea-
tures. In the spatial domain, Wang et al. [51] demonstrated
that genuine images exhibit higher denoising diffusion im-
plicit models (DDIM) inversion errors than AI-generated
images. In the frequency domain, Jeong et al. [16–18] ex-
plored techniques leveraging frequency-based artifacts for
improved detection. Meanwhile, multimodal methods have
shown promise in enhancing robustness and generalization
across diverse datasets. For instance, Chang et al. [5] intro-
duced a zero-shot deepfake detection method using VLMs
like InstructBLIP. By framing detection as a visual question
answering (VQA) task, they improved accuracy on unseen
data through prompt tuning. Similarly, Keita et al. [22] pro-
posed Bi-LORA, which reframes binary classification as an
image captioning problem, leading to high-precision syn-
thetic image detection. Other studies have focused on in-
tegrating category-specific information to enhance feature
extraction. Tan et al. [46] developed C2P-CLIP, which
embeds category-specific concepts into CLIP’s image en-
coder, achieving state-of-the-art generalization in deepfake
detection. Huang et al. [13] introduced OW-FFA-VQA,
leveraging a VQA framework and the FFA-VQA dataset
for explainable face forgery analysis using a fine-tuned
multimodal large language model (MLLM) and multimodal
image detection system (MIDS). Furthermore, Iliopoulou
et al. [14] explored compression-based detection methods
using a variational autoencoder (V AE), distinguishing real
and synthetic faces based on reconstruction quality. Kout-
liset al. [23] further improved synthetic image detection by
mapping intermediate CLIP Transformer block features to
a forgery-aware vector space, achieving a +10.6% perfor-
mance improvement over previous state-of-the-art methods
with minimal training.
3

4-class setting
ProGAN
+
Embeddinglabel
Vision
EncoderVision
EncoderPerceiver
  ResamplerPerceiver
  Resampler
1st GA TED XA TTN-DENSEn-th LM block
n-th GA TED XA TTN-DENSE
1st LM block
Processed text
Interleaved visual/text data
It is real. It is      < image > It is real. < image > It is/Output:  text
LoRA
RAVID_ CLIPLoRA
RAVID_ CLIP
1Query
2Relevant InfoFigure 2. Our RA VID integrates RA VID CLIP for embedding-based image retrieval and Openflamingo for decision-making: (1) 4-class
ProGAN training set images are encoded into vector embeddings using RA VID CLIP and stored in a Milvus vector database; (2) At testing
time, the query image embedding is matched against stored embeddings to retrieve the most relevant images; (3) The retrieved images and
labels serve as contextual information, combined with the query image, and processed by Openflamingo.
Despite the substantial progress in AI-generated image
detection, none of the existing methods have explored the
integration of RAG within VLMs for this task. Given
RAG’s proven success in various domains, its potential
for enhancing AI-generated image detection remains en-
tirely unexplored. Addressing this critical gap, we intro-
duce RA VID, the first RAG-driven VLM framework for
AI-generated image detection, leveraging retrieved visual
knowledge to improve detection accuracy, robustness, and
generalization across diverse generative models.
3. Proposed Method
In this section, we present RA VID, which performs re-
trieval of query-relevant images over vector image corpus as
the external knowledge source and determines query-image
class (label) grounded in them.
3.1. Motivation
As generative models continue to improve, distinguish-
ing AI-generated images from real ones has become in-
creasingly challenging. Traditional detection methods of-
ten rely on low-level artifacts or model-specific fingerprints,
making them prone to overfitting and inconsistent perfor-
mance across different synthesis techniques. These meth-
ods also struggle with generalization and lack robustness,
as even small changes can significantly impact detection ac-
curacy. Recent research often focuses on using VLMs for
AI-generated image detection. However, most of these ap-
proaches require extensive fine-tuning of VLMs, which is
computationally expensive, limits scalability, and can lead
to overfitting on specific datasets.
To address these challenges, we propose RA VID, a novel
approach that combines RAG with AI-generated image de-
tection. By integrating external image retrieval, RA VIDdynamically enhances detection by providing relevant con-
text, overcoming the limitations of existing methods with-
out the need for VLM fine-tuning. Our approach improves
the robustness of AI-generated image detection, ensuring
higher accuracy and better generalization in diverse scenar-
ios by incorporating large-scale visual knowledge through
a vector image database. With RA VID, we present a novel
approach that combines powerful image embeddings with
visual-language models, without the need for fine-tuning.
This method sets a new standard for detecting AI-generated
images, offering improved performance across a wide vari-
ety of generative models.
3.2. Concept-Aware Image Embeddings
Recent work by Tan et al. [46] demonstrates that CLIP fea-
tures’ ability to detect AI-generated images through linear
classification is largely due to their capacity to capture con-
ceptual similarities . Building on this insight, they propose
an approach that incorporates enhanced captions and con-
trastive learning to embed categorical concepts into the im-
age encoder, thereby improving the distinction between real
and generated images. Inspired by this work, we adopt a
similar strategy in our framework to generate high-quality
embeddings for a vector database, which is crucial to the
performance of the RA VID method.
1.Caption Generation and Enhancement :
LetDrepresent the training dataset containing both real
and synthetic images, defined as D={(xj, yj)}N
j=1,
where yj∈ {0,1}indicates whether the image xjis real
(y= 0) or fake ( y= 1). For each image in the dataset,
we generate captions using the ClipCap model [29], re-
sulting in a set of captions C={(cj, yj)}N
j=1.
To enhance these captions, we append category-specific
4

80
 60
 40
 20
 020 40 60 8060
40
20
020406080Progan
Real
Fake
60
 40
 20
 0 20 40 60 8080
60
40
20
020406080Stylegan
Real
Fake
60
 40
 20
 0 20 4040
20
02040Stargan
Real
Fake
75
 50
 25
 0 25 50 7580
60
40
20
020406080Crn
Real
Fake
80
 60
 40
 20
 020 40 60 8080
60
40
20
020406080Imle
Real
Fake
60
 40
 20
 0 20 40 6040
20
02040Guided
Real
Fake
60
 40
 20
 0 20 40 6030
20
10
0102030Glide_100_10
Real
Fake
60
 40
 20
 0 20 40 6030
20
10
0102030Ldm_200
Real
FakeFigure 3. t-SNE Visualization of RA VID CLIP’s Feature Distributions for Different Generative Models. The scatter plots illustrate the
t-SNE embeddings of features extracted from real (green) and generated (red) images across various generative models.
prompts P={Preal, Pfake}to the original captions, as
proposed by [46]. Specifically, the enhanced captions
˜C={˜cj}N
j=1are defined as:
˜cj=(
(Preal, cj),ifyj= 0
(Pfake, cj),ifyj= 1(1)
In this formulation, Preal and Pfake represent
category-specific prompts (e.g., Preal = Camera,
Pfake=Deepfake) that provide additional context to
differentiate real images from synthetic ones.
2.Concept Injection via Contrastive Learning :
To integrate classification concepts into the image en-
coder, we employ a contrastive learning framework. In
this approach, the text encoder remains frozen, while
Low-Rank Adaptation (LoRA) layers are applied to the
image encoder to facilitate learning. Given an image xj
and its corresponding enhanced caption ˜cj, their feature
representations are computed as follows:
tj=encoder text(˜cj),ej=encoder img(xj), (2)
where tjand ejdenote the text and image embeddings,
respectively.
To ensure that the image encoder aligns visual features
with their corresponding textual descriptions, we opti-
mize a contrastive loss function Lcontrastive , defined as:
Lcontrastive =1
2(Le→t+Lt→e), (3)
where Le→tenforces alignment between image embed-
dings and their respective text embeddings, while Lt→eensures reverse alignment. These losses are formulated
as:
Le→t=−1
NNX
i=1logexp( eT
i·ti)PN
j=1exp( eT
i·tj), (4)
Lt→e=−1
NNX
i=1logexp(tT
i·ei)PN
j=1exp(tT
i·ej). (5)
Here, eT
itjrepresents the dot product between the im-
age feature eiand the text feature tj, capturing their
similarity. The denominator normalizes the similarity
scores across all samples in the batch, ensuring a well-
structured representation space.
By optimizing Lcontrastive , the image encoder learns to
map visual features into a space that aligns with their tex-
tual descriptions. This process effectively injects classi-
fication concepts within the image encoder, enhancing
its ability to distinguish between real and AI-generated
images.
The CLIP concept-injection-enhanced model is then used
to generate embeddings for both real and fake images.
These embeddings are stored in a vector database (e.g., Mil-
vus [49]), which serves as an external knowledge source for
the RA VID framework.
3.3. Image Retrieval
Retrieval involves computing the similarity between the
query image qimgand each knowledge element (image em-
beddings) to determine relevance. To achieve this, we first
embed the query image qimgusing the RA VID CLIP image
encoder to obtain its embedding fembed . Relevance is then
computed based on representation-level similarity, such as
5

LLM Prompt for AI-Generated Image Detection
Role: You are an AI-generated image Detection Sys-
tem that leverages visual retrieval-augmented genera-
tion to enhance accuracy and robustness through mul-
timodal context fusion.
Task: Identify whether a given image is AI-generated
or real by retrieving relevant visual references and ana-
lyzing them alongside the query image using a vision-
language model.
Objective: Achieve SOTA performance in AI-
generated image detection with strong generalization
across diverse generative models and high robustness
under image degradations.
Constraints:
• Answer the question with a single word.
Search Space: Use only knowledge from the addi-
tional context in decision-making.
Figure 4. Example of initial prompt used to guide LLM architec-
ture generation.
cosine similarity, to measure the alignment between the
query embedding and stored embeddings in the external
corpus C. The retrieval process is formulated as:
I=argmaxk
Ii∈Csim(fembed, Ei), (6)
where fembed is the embedding of the query image qimg
computed by the RA VID CLIP image encoder, Eirepre-
sents the stored embedding of an image Iiin the external
corpus C, and sim (fembed, Ei)denotes the cosine similarity
between the query embedding and each corpus embedding,
computed as:
sim(fembed, Ei) =fembed·Ei
∥fembed∥∥Ei∥(7)
where argmaxk
Ii∈Cselects the top- kimages with the highest
similarity scores.
By retrieving the top- kmost relevant images, this ap-
proach ensures that the subsequent answer generation step
benefits from rich contextual information, improving the ro-
bustness of AI-generated image detection.
3.4. Image-Augmented Response Generation
After retrieving the most relevant images, the next step
is to integrate them into the response generation process
to enhance the quality and contextual accuracy of the
generated output. To achieve this, we first construct a
multimodal context by pairing each retrieved image with
its corresponding label. These multimodal pairs are then
concatenated to form a comprehensive context represen-
tation. Finally, the query image is incorporated into this
structured input, which serves as the input to a VLM, such
as Openflamingo. Formally, this process is represented as:[
{’text’: ’Is this photo real? Please provide
your answer. You should ONLY output "real
" or "fake".’},
{’image’: ’path to img_1’},
{’text’: ’User: It is \nAssistant:
img_1_label’},
{’image’: ’path to img_2’},
{’text’: ’User: It is \nAssistant:
img_2_label’},
...
{’image’: ’path to img_n’},
{’text’: ’User: It is \nAssistant:
img_n_label’},
{’image’: ’path to q_img’},
{’text’: ’User: It is \nAssistant: ’}
]
This structured input is then fed into the VLM, which
jointly processes visual, textual, and query-specific infor-
mation. By leveraging this multimodal richness, the model
generates a response that effectively integrates retrieved
knowledge to improve AI-generated image detection accu-
racy.
4. Experimental Results
In this section, we present an extensive evaluation covering
multiple aspects, such as datasets, implementation details,
and AI-generated image detection performance.
4.1. Experimental Setup
Dataset. To ensure a fair comparison, we utilize the widely
recognized UniversalFakeDetect dataset [32], which has
been extensively used in prior benchmarks. This allows
for a direct evaluation of RA VID against state-of-the-art
methods, ensuring consistency and robustness in perfor-
mance assessment. Following the experimental setup in-
troduced by Wang et al. [50], the dataset employs Pro-
GAN as the training set, comprising 20 subsets of gen-
erated images. For constructing our vector database, we
adopt a 4-class setting (horse, chair, cat, car) as outlined
by Tan et al. [46]. The test set consists of 19 subsets
generated by a diverse range of models, including Pro-
GAN [19], StyleGAN [20], BigGAN [3], CycleGAN [54],
StarGAN [8], GauGAN [34], Deepfake [41], CRN [7],
IMLE [25], SAN [10], SITD [6], Guided Diffusion [11],
LDM [40], GLIDE [31], and DALLE [37].
Evaluation Metrics. Following the convention of previous
detection methods [21, 32, 46, 51], we report the accuracy
(ACC). We also calculate the mean accuracy across all data
subsets to provide a more comprehensive evaluation of
overall model performance.
Baselines. In our study, we fine-tuned AntiFakePrompt [5]
and Bi-LORA [21, 22]. Moreover, we have chosen the
6

Table 1. Accuracy (ACC) scores of state-of-art detectors and RA VID across 19 test datasets. Best performance is denoted with bold . We
report the results of the best RA VID models with different VLMs.
Methods RefGAN Deep
FakesLow level Perceptual lossGuidedLDM GlideDalle mAcc
Pro-
GANCycle-
GANBig-
GANStyle-
GANGau-
GANStar-
GANSITD SAN CRN IMLE200
steps200
w/cfg100
steps100
2750
27100
10
Freq-spec WIFS2019 49.90 99.90 50.50 49.90 50.30 99.70 50.10 50.00 48.00 50.60 50.10 50.90 50.40 50.40 50.30 51.70 51.40 50.40 50.00 55.45
Co-occurence Elect. Imag. 97.70 97.70 53.75 92.50 51.10 54.70 57.10 63.06 55.85 65.65 65.80 60.50 70.70 70.55 71.00 70.25 69.60 69.90 67.55 66.86
CNN-Spot CVPR2020 99.99 85.20 70.20 85.70 78.95 91.70 53.47 66.67 48.69 86.31 86.26 60.07 54.03 54.96 54.14 60.78 63.80 65.66 55.58 69.58
Patchfor ECCV2020 75.03 68.97 68.47 79.16 64.23 63.94 75.54 75.14 75.28 72.33 55.30 67.41 76.50 76.10 75.77 74.81 73.28 68.52 67.91 71.24
F3Net ECCV2020 99.38 76.38 65.33 92.56 58.10 100.00 63.48 54.17 47.26 51.47 51.47 69.20 68.15 75.35 68.80 81.65 83.25 83.05 66.30 71.33
Bi-LORA ICASSP2023 98.71 96.74 81.18 78.30 96.30 86.32 57.78 68.89 52.28 73.00 82.60 65.10 85.15 59.20 85.00 83.50 85.65 84.90 72.70 78.59
LGrad CVPR2023 99.84 85.39 82.88 94.83 72.45 99.62 58.00 62.50 50.00 50.74 50.78 77.50 94.20 95.85 94.80 87.40 90.70 89.55 88.35 80.28
UniFD CVPR2023 100.00 98.50 94.50 82.00 99.50 97.00 66.60 63.00 57.50 59.50 72.00 70.03 94.19 73.76 94.36 79.07 79.85 78.14 86.78 81.38
AntiFakePrompt CVPR2023 99.26 96.82 87.88 80.00 98.13 83.57 60.20 70.56 53.70 79.21 79.01 73.75 89.55 64.10 89.80 93.55 93.90 92.95 80.10 82.42
FreqNet AAAI2024 97.90 95.84 90.45 97.55 90.24 93.41 97.40 88.92 59.04 71.92 67.35 86.70 84.55 99.58 65.56 85.69 97.40 88.15 59.06 85.09
NPR CVPR2024 99.84 95.00 87.55 96.23 86.57 99.75 76.89 66.94 98.63 50.00 50.00 84.55 97.65 98.00 98.20 96.25 97.15 97.35 87.15 87.56
FatFormer CVPR2024 99.89 99.32 99.50 97.15 99.41 99.75 93.23 81.11 68.04 69.45 69.45 76.00 98.60 94.90 98.65 94.35 94.65 94.20 98.75 90.86
RINE ECCV2024 100.00 99.30 99.60 88.90 99.80 99.50 80.60 90.60 68.30 89.20 90.60 76.10 98.30 88.20 98.60 88.90 92.60 90.70 95.00 91.31
C2P-CLIP⋆Arxiv 2024 99.71 90.69 95.28 99.38 95.26 96.60 89.86 98.33 64.61 90.69 90.69 77.80 99.05 98.05 98.95 94.65 94.20 94.40 98.80 93.00
C2P-CLIP‡Arxiv 2024 99.98 97.31 99.12 96.44 99.17 99.60 93.77 95.56 64.38 93.29 93.29 69.10 99.25 97.25 99.30 95.25 95.25 96.10 98.55 93.79
RA VID (N=13) Gemma3 97.34 92.73 92.92 89.68 95.55 93.42 76.11 72.22 62.33 88.62 88.37 67.25 95.60 92.45 95.55 92.50 92.90 93.30 93.60 88.02
RA VID (N=13) Qwen-VL 99.96 97.84 98.70 95.24 99.28 99.82 93.36 93.61 63.01 96.47 96.46 67.80 99.30 96.75 99.40 94.05 95.00 95.70 98.40 93.69
RA VID (N=13) Openflamingo 99.98 97.35 99.15 96.27 99.33 99.82 93.47 95.00 63.70 95.53 95.56 68.75 99.20 96.95 99.35 94.65 94.90 95.85 98.35 93.85
(⋆) Trump,Biden. ( ‡) Deepfake,Camera.
latest and most competitive methods in the field, including
Co-occurence [30], Freq-spec [53], CNN-Spot [50], Fatch-
For [4], UniFD [32], LGrad [45], F3Net [36], FreqNet [47],
NPR [48], Fatformer [27], C2P-CLIP [46], RINE [23], for
all those models, we report the results presented in [46].
Finally, for RINE, we report results from its paper [23].
Implementation Details. To fine-tune CLIP ViT-L/14, we
use Adam optimizer with an initial learning rate of 4×10−4,
a batch size of 128, and train for 1 epoch. We apply LoRA
layers to the qproj ,kproj , and vproj layers using the
Parameter-Efficient Fine-Tuning (PEFT) library. The LoRA
hyper-parameters are set as follows: lorar= 6,loraalpha
= 6, and loradropout = 0.8. For the vector database,
we use Milvus locally via Docker. On the other hand,
for image-augmented response generation, we use Open-
flamingo [2].
4.2. Comparison with the State-of-the-art
Table 1 presents the mean accuracy (mAcc) scores
for cross-generator detection on the UniversalFakeDetect
dataset, which includes 19 different generative models
spanning GANs, Deepfakes, low-level vision models, per-
ceptual loss models, and diffusion models. RA VID achieves
93.85% mAcc, outperforming 15 state-of-the-art methods
and demonstrating strong generalization across diverse im-
age synthesis techniques.
The baseline methods, such as Freq-spec and CNN-
Spot, rely on traditional frequency analysis and convolu-
tional neural networks for detecting AI-generated images.
In contrast, RA VID adopts a novel approach to enhance
feature representation, ensuring better generalization across
various generators. Compared to UniFD, a recent state-
of-the-art method, RA VID improves the mAcc by 12.47%,
highlighting the effectiveness of our approach. Addition-ally, RA VID demonstrates competitive performance with
the latest methods, RINE and C2P-CLIP, achieving a mere
1.48% and 0.16% mAcc gap, respectively. While RINE
utilizes advanced feature extraction and fusion techniques,
which increase computational complexity, C2P-CLIP em-
beds category-specific concepts into CLIP’s image encoder.
Meanwhile, our method strikes a balance between perfor-
mance and efficiency, making it more suitable for real-
world applications.
We also analyze detection performance across different
generator categories. For Perceptual Loss models, RA VID
achieves an impressive detection accuracy of 95.56%, while
for Low-level Vision models, it achieves 95.00%, demon-
strating competitive performance with all other methods in
these categories. This demonstrates RA VID’s ability to ef-
fectively handle challenging generators without requiring
specialized architectural modifications or additional com-
putational resources.
4.3. Impact of Retrieved Image Count on Detection
Performance
To evaluate the impact of the number of retrieved images
on RA VID’s AI-generated image detection performance,
we conducted experiments with varying retrieval settings.
Specifically, we compared performance when retrieving 1
(N= 1 ), 3 ( N= 3 ), 5 ( N= 5 ), 7 ( N= 7 ), and
13 (N= 13 ) images, utilizing the Openflamingo vision-
language model. The results, presented in Table 5, show a
substantial improvement in detection accuracy as the num-
ber of retrieved images increases. With one retrieved image,
the model achieved an average accuracy (mAcc) of 50.00%,
whereas increasing the retrieval count to N= 3led to a sig-
nificant boost to 93.53%. This improvement was consistent
across various generative models, such as ProGAN (50.00%
→99.95%), StarGAN (50.00% →99.85%), and Deep-
7

Table 2. In-context learning without RAG. Instead of retrieving relevant images to the query image, we randomly select Nimages from
the 4-class setting ProGAN training set. Best performance is denoted with bold .
Methods VLMs N ShotsGAN Deep
FakesLow level Perceptual lossGuidedLDM GlideDalle mAcc
Pro-
GANCycle-
GANBig-
GANStyle-
GANGau-
GANStar-
GANSITD SAN CRN IMLE200
steps200
w/cfg100
steps100
2750
27100
10
RA VID W/ RAG Openflamingo 3 99.95 97.84 99.25 95.94 99.38 99.85 92.64 93.61 62.33 97.55 97.59 66.95 99.35 96.35 99.35 93.10 93.25 94.40 98.40 93.53
RA VID W/ RAG Openflamingo 13 99.98 97.35 99.15 96.27 99.33 99.82 93.47 95.00 63.70 95.53 95.56 68.75 99.20 96.95 99.35 94.65 94.90 95.85 98.35 93.85
RA VID W/O RAG Openflamingo 3 49.49 50.53 51.00 49.99 49.64 50.88 50.53 49.17 45.89 49.89 49.87 51.40 50.65 50.40 50.35 50.30 50.45 50.95 50.45 50.10
RA VID W/O RAG Openflamingo 13 49.38 51.59 50.78 49.67 49.80 50.55 51.14 50.00 47.72 50.65 49.85 48.50 49.25 49.85 48.90 49.70 50.50 51.55 48.80 49.90
Table 3. Comparison of image retrieval performance in RA VID: Fine-Tuned vs. Non-Fine-Tuned CLIP as an image embedding model for
retrieving relevant images in the context of AI-generated images. The notation ( *) denotes RA VID using a fine-tuned CLIP model as the
image embedding method.
Methods VLMs N ShotsGAN Deep
FakesLow level Perceptual lossGuidedLDM GlideDalle mAcc
Pro-
GANCycle-
GANBig-
GANStyle-
GANGau-
GANStar-
GANSITD SAN CRN IMLE200
steps200
w/cfg100
steps100
2750
27100
10
RA VID W/ RAG (*) Openflamingo 1 50.00 50.00 50.00 50.00 50.00 50.00 50.08 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00
RA VID W/ RAG (*) Openflamingo 3 99.95 97.84 99.25 95.94 99.38 99.85 92.64 93.61 62.33 97.55 97.59 66.95 99.35 96.35 99.35 93.10 93.25 94.40 98.40 93.53
RA VID W/ RAG (*) Openflamingo 5 99.95 97.58 99.28 96.24 99.30 99.82 93.19 93.89 63.24 96.26 96.25 68.10 99.20 96.45 99.30 94.00 94.35 95.10 98.30 93.67
RA VID W/ RAG (*) Openflamingo 7 99.96 97.46 99.15 96.24 99.32 99.80 93.30 94.72 63.93 96.04 96.03 68.30 99.25 96.65 99.35 94.00 94.70 95.30 98.15 93.77
RA VID W/ RAG (*) Openflamingo 13 99.98 97.35 99.15 96.27 99.33 99.82 93.47 95.00 63.70 95.53 95.56 68.75 99.20 96.95 99.35 94.65 94.90 95.85 98.35 93.85
RA VID W/ RAG Openflamingo 1 50.00 50.00 50.00 50.00 50.00 50.00 50.08 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00
RA VID W/ RAG Openflamingo 3 97.76 89.06 86.93 76.91 95.34 98.32 70.53 70.83 56.85 67.11 76.90 69.30 85.65 67.30 85.85 80.20 81.05 82.00 78.15 79.79
RA VID W/ RAG Openflamingo 5 98.09 89.74 88.20 77.59 95.30 98.62 73.08 65.28 59.13 71.01 79.20 71.55 86.90 69.90 88.25 84.90 85.10 84.50 80.80 81.43
RA VID W/ RAG Openflamingo 7 98.16 89.86 88.00 78.13 95.06 98.57 74.04 65.83 61.42 69.12 77.94 71.75 87.60 71.75 88.35 86.10 86.70 86.00 82.35 81.93
RA VID W/ RAG Openflamingo 13 98.23 88.72 88.13 77.61 94.68 98.37 75.30 69.17 58.90 68.10 76.32 71.75 87.70 71.40 88.40 86.30 87.40 86.70 82.45 81.88
Fakes (50.08% →92.64%). As the number of retrieved
images increases to N=5,N=7 and N=13, mAcc in-
creases progressively to 93.67%, 93.77% and 93.85%, re-
spectively. These findings indicate that incorporating addi-
tional retrieved images provides richer contextual informa-
tion, thereby improving the model’s generalization across
diverse AI-generated image models.
Figure 6 RA VID’s detection performance across differ-
ent categories of AI-generated images, with varying num-
bers of retrieved images (shots). As the number of retrieved
images increases from 1 to 3, a substantial improvement in
detection accuracy is observed for several generative mod-
els. This indicates that incorporating more contextual infor-
mation enhances RA VID’s ability to identify AI-generated
content. However, the performance gains become mini-
mal as the number of retrieved images exceeds 3. This
highlights how additional context enhances RA VID’s ro-
bustness in distinguishing AI-generated images. Overall,
these results underscore the value of leveraging retrieval-
augmented generation for improved detection performance.
4.4. Impact of Using RAG for Retrieval in RA VID
To evaluate the impact of dynamically retrieving relevant
images in the RA VID approach, we conducted an experi-
ment where the context provided to the VLM was formed
by randomly selecting images, instead of using the RAG re-
trieval mechanism. This experiment allowed us to assess
the significance of the retrieval process in improving detec-
tion accuracy. In this setup, rather than retrieving relevant
images related to the query image, we randomly selected N
images from the 4-class setting ProGAN training set. These
randomly chosen images, along with their corresponding la-bels, were used as a context for the detection task. This
mimicked the in-context learning strategy used in RAG, but
without its retrieval component. We maintained the same
configurations as in RA VID, varying the number of selected
images (shots). In the 3-shot setup, three randomly selected
image was provided as context, while in the 13-shot setup,
thirteen images were used as context.
The results in Table 2 show the detection accuracy
(mAcc ) across a range of generative models. For the 3-
shot setup, RA VID W/O RAG achieved an average accu-
racy of 50.10%, whereas in the 13-shot setup, the accu-
racy slightly declined to 49.90%. While these results in-
dicate that providing more context does not improve per-
formance, they still fall behind the accuracy achieved when
using RAG, where the retrieved context is more relevant to
the query image. This experiment highlights the importance
of relevant context in AI-generated image detection. When
the model relies on randomly selected images, the context
lacks meaningful relevance to the query, limiting its abil-
ity to make accurate predictions, especially with complex
generative models. In contrast, the ability of RAG to re-
trieve relevant images substantially boosts the model’s per-
formance, emphasizing the critical role of relevant context
in improving detection accuracy and generalization. This
investigation also underscores the need for an image em-
bedding model for retrieval that is sensitive to the subtle
characteristics of AI-generated images, as opposed to one
that focuses on general cues, which are less useful for this
task.
To quantify the impact of retrieval, we analyze, the accu-
racy gap between RAG-based and non-RAG-based setups.
This difference, mAcc(W/ RAG) – mAcc(W/O RAG), di-
8

/uni00000053/uni00000055/uni00000052/uni0000004a/uni00000044/uni00000051/uni00000046/uni0000005c/uni00000046/uni0000004f/uni00000048/uni0000004a/uni00000044/uni00000051/uni00000045/uni0000004c/uni0000004a/uni0000004a/uni00000044/uni00000051/uni00000056/uni00000057/uni0000005c/uni0000004f/uni00000048/uni0000004a/uni00000044/uni00000051/uni0000004a/uni00000044/uni00000058/uni0000004a/uni00000044/uni00000051/uni00000056/uni00000057/uni00000044/uni00000055/uni0000004a/uni00000044/uni00000051
/uni00000047/uni00000048/uni00000048/uni00000053/uni00000049/uni00000044/uni0000004e/uni00000048
/uni00000056/uni00000048/uni00000048/uni0000004c/uni00000051/uni0000004a/uni00000047/uni00000044/uni00000055/uni0000004e
/uni00000056/uni00000044/uni00000051
/uni00000046/uni00000055/uni00000051
/uni0000004c/uni00000050/uni0000004f/uni00000048
/uni0000004a/uni00000058/uni0000004c/uni00000047/uni00000048/uni00000047
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000015/uni00000013/uni00000013
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000015/uni00000013/uni00000013/uni00000042/uni00000046/uni00000049/uni0000004a
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000014/uni00000013/uni00000013/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000018/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000014/uni00000013/uni00000047/uni00000044/uni0000004f/uni0000004f/uni00000048/uni00000031/uni00000003/uni00000020/uni00000003/uni00000014
/uni00000053/uni00000055/uni00000052/uni0000004a/uni00000044/uni00000051/uni00000046/uni0000005c/uni00000046/uni0000004f/uni00000048/uni0000004a/uni00000044/uni00000051/uni00000045/uni0000004c/uni0000004a/uni0000004a/uni00000044/uni00000051/uni00000056/uni00000057/uni0000005c/uni0000004f/uni00000048/uni0000004a/uni00000044/uni00000051/uni0000004a/uni00000044/uni00000058/uni0000004a/uni00000044/uni00000051/uni00000056/uni00000057/uni00000044/uni00000055/uni0000004a/uni00000044/uni00000051
/uni00000047/uni00000048/uni00000048/uni00000053/uni00000049/uni00000044/uni0000004e/uni00000048
/uni00000056/uni00000048/uni00000048/uni0000004c/uni00000051/uni0000004a/uni00000047/uni00000044/uni00000055/uni0000004e
/uni00000056/uni00000044/uni00000051
/uni00000046/uni00000055/uni00000051
/uni0000004c/uni00000050/uni0000004f/uni00000048
/uni0000004a/uni00000058/uni0000004c/uni00000047/uni00000048/uni00000047
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000015/uni00000013/uni00000013
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000015/uni00000013/uni00000013/uni00000042/uni00000046/uni00000049/uni0000004a
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000014/uni00000013/uni00000013/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000018/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000014/uni00000013/uni00000047/uni00000044/uni0000004f/uni0000004f/uni00000048/uni00000031/uni00000003/uni00000020/uni00000003/uni00000016
/uni00000053/uni00000055/uni00000052/uni0000004a/uni00000044/uni00000051/uni00000046/uni0000005c/uni00000046/uni0000004f/uni00000048/uni0000004a/uni00000044/uni00000051/uni00000045/uni0000004c/uni0000004a/uni0000004a/uni00000044/uni00000051/uni00000056/uni00000057/uni0000005c/uni0000004f/uni00000048/uni0000004a/uni00000044/uni00000051/uni0000004a/uni00000044/uni00000058/uni0000004a/uni00000044/uni00000051/uni00000056/uni00000057/uni00000044/uni00000055/uni0000004a/uni00000044/uni00000051
/uni00000047/uni00000048/uni00000048/uni00000053/uni00000049/uni00000044/uni0000004e/uni00000048
/uni00000056/uni00000048/uni00000048/uni0000004c/uni00000051/uni0000004a/uni00000047/uni00000044/uni00000055/uni0000004e
/uni00000056/uni00000044/uni00000051
/uni00000046/uni00000055/uni00000051
/uni0000004c/uni00000050/uni0000004f/uni00000048
/uni0000004a/uni00000058/uni0000004c/uni00000047/uni00000048/uni00000047
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000015/uni00000013/uni00000013
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000015/uni00000013/uni00000013/uni00000042/uni00000046/uni00000049/uni0000004a
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000014/uni00000013/uni00000013/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000018/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000014/uni00000013/uni00000047/uni00000044/uni0000004f/uni0000004f/uni00000048/uni00000031/uni00000003/uni00000020/uni00000003/uni00000018
/uni00000053/uni00000055/uni00000052/uni0000004a/uni00000044/uni00000051/uni00000046/uni0000005c/uni00000046/uni0000004f/uni00000048/uni0000004a/uni00000044/uni00000051/uni00000045/uni0000004c/uni0000004a/uni0000004a/uni00000044/uni00000051/uni00000056/uni00000057/uni0000005c/uni0000004f/uni00000048/uni0000004a/uni00000044/uni00000051/uni0000004a/uni00000044/uni00000058/uni0000004a/uni00000044/uni00000051/uni00000056/uni00000057/uni00000044/uni00000055/uni0000004a/uni00000044/uni00000051
/uni00000047/uni00000048/uni00000048/uni00000053/uni00000049/uni00000044/uni0000004e/uni00000048
/uni00000056/uni00000048/uni00000048/uni0000004c/uni00000051/uni0000004a/uni00000047/uni00000044/uni00000055/uni0000004e
/uni00000056/uni00000044/uni00000051
/uni00000046/uni00000055/uni00000051
/uni0000004c/uni00000050/uni0000004f/uni00000048
/uni0000004a/uni00000058/uni0000004c/uni00000047/uni00000048/uni00000047
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000015/uni00000013/uni00000013
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000015/uni00000013/uni00000013/uni00000042/uni00000046/uni00000049/uni0000004a
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000014/uni00000013/uni00000013/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000018/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000014/uni00000013/uni00000047/uni00000044/uni0000004f/uni0000004f/uni00000048/uni00000031/uni00000003/uni00000020/uni00000003/uni0000001a
/uni00000053/uni00000055/uni00000052/uni0000004a/uni00000044/uni00000051/uni00000046/uni0000005c/uni00000046/uni0000004f/uni00000048/uni0000004a/uni00000044/uni00000051/uni00000045/uni0000004c/uni0000004a/uni0000004a/uni00000044/uni00000051/uni00000056/uni00000057/uni0000005c/uni0000004f/uni00000048/uni0000004a/uni00000044/uni00000051/uni0000004a/uni00000044/uni00000058/uni0000004a/uni00000044/uni00000051/uni00000056/uni00000057/uni00000044/uni00000055/uni0000004a/uni00000044/uni00000051
/uni00000047/uni00000048/uni00000048/uni00000053/uni00000049/uni00000044/uni0000004e/uni00000048
/uni00000056/uni00000048/uni00000048/uni0000004c/uni00000051/uni0000004a/uni00000047/uni00000044/uni00000055/uni0000004e
/uni00000056/uni00000044/uni00000051
/uni00000046/uni00000055/uni00000051
/uni0000004c/uni00000050/uni0000004f/uni00000048
/uni0000004a/uni00000058/uni0000004c/uni00000047/uni00000048/uni00000047
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000015/uni00000013/uni00000013
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000015/uni00000013/uni00000013/uni00000042/uni00000046/uni00000049/uni0000004a
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000014/uni00000013/uni00000013/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000018/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000014/uni00000013/uni00000047/uni00000044/uni0000004f/uni0000004f/uni00000048/uni00000031/uni00000003/uni00000020/uni00000003/uni00000014/uni00000016Figure 5. Performance comparison of RA VID W/ CLIP fine-tuned across different N-shot settings.
/uni00000033/uni00000055/uni00000052/uni00000010/uni0000002a/uni00000024/uni00000031/uni00000026/uni0000005c/uni00000046/uni0000004f/uni00000048/uni00000010/uni0000002a/uni00000024/uni00000031/uni00000025/uni0000004c/uni0000004a/uni00000010/uni0000002a/uni00000024/uni00000031/uni00000036/uni00000057/uni0000005c/uni0000004f/uni00000048/uni00000010/uni0000002a/uni00000024/uni00000031/uni0000002a/uni00000044/uni00000058/uni00000010/uni0000002a/uni00000024/uni00000031/uni00000036/uni00000057/uni00000044/uni00000055/uni00000010/uni0000002a/uni00000024/uni00000031
/uni00000027/uni00000048/uni00000048/uni00000053/uni00000029/uni00000044/uni0000004e/uni00000048/uni00000056
/uni00000036/uni0000002c/uni00000037/uni00000027
/uni00000036/uni00000024/uni00000031
/uni00000026/uni00000035/uni00000031
/uni0000002c/uni00000030/uni0000002f/uni00000028
/uni0000002a/uni00000058/uni0000004c/uni00000047/uni00000048/uni00000047
/uni0000002f/uni00000027/uni00000030/uni00000042/uni00000015/uni00000013/uni00000013/uni00000042/uni00000056/uni00000057/uni00000048/uni00000053/uni00000056
/uni0000002f/uni00000027/uni00000030/uni00000042/uni00000015/uni00000013/uni00000013/uni00000042/uni0000005a/uni00000012/uni00000046/uni00000049/uni0000004a
/uni0000002f/uni00000027/uni00000030/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000056/uni00000057/uni00000048/uni00000053/uni00000056
/uni0000002a/uni0000002f/uni0000002c/uni00000027/uni00000028/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000002a/uni0000002f/uni0000002c/uni00000027/uni00000028/uni00000042/uni00000018/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000002a/uni0000002f/uni0000002c/uni00000027/uni00000028/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000014/uni00000013/uni0000002a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000027/uni00000044/uni0000004f/uni0000004f/uni00000048/uni0000003a/uni00000012/uni00000003/uni00000035/uni00000024/uni0000002a/uni0000000f/uni00000003/uni00000031/uni00000020/uni00000016
/uni0000003a/uni00000012/uni00000003/uni00000035/uni00000024/uni0000002a/uni0000000f/uni00000003/uni00000031/uni00000020/uni00000014/uni00000016/uni0000003a/uni00000012/uni00000032/uni00000003/uni00000035/uni00000024/uni0000002a/uni0000000f/uni00000003/uni00000031/uni00000020/uni00000016
/uni0000003a/uni00000012/uni00000032/uni00000003/uni00000035/uni00000024/uni0000002a/uni0000000f/uni00000003/uni00000031/uni00000020/uni00000014/uni00000016
Figure 6. Performance comparison of RA VID W/ RAG and W/O
RAG across different N-shot settings. The accuracy shift demon-
strates the positive impact of retrieval, with higher performance in
the W/RAG setup.
rectly measures the benefit of retrieving relevant images.
Specifically, when RA VID uses RAG with three retrieval
shot, the mean accuracy ( mAcc ) improves from 50.10%
(W/O RAG) to 93.53% (W/ RAG), representing a signifi-
cant increase of 43.43%. And if we increase the retrieval
shot to 13, the average accuracy rises from 49.90%(W/O
RAG ) to 93.85% (W/ RAG), representing an increase of
43.95%. As illustrated in Figure 6, this significant im-
provement is consistently positive, confirming that retrieval
improves AI-generated image detection. The consistently
higher accuracy of W/ RAG over W/O RAG underscores the
importance of meaningful context. Without retrieval, the
model relies on irrelevant information, leading to weaker
feature alignment and lower accuracy. A well-designed re-
trieval mechanism ensures that retrieved images share key
characteristics with the query, enhancing generalization and
improving detection performance.4.5. Relevance of CLIP Task-Tuning for Contextual
Image Retrieval
To assess the impact of fine-tuning the CLIP image en-
coder on retrieving more relevant context, we compare two
configurations: pretrained CLIP (without fine-tuning) and
fine-tuned RA VID CLIP. In both cases, retrieved images
are used as context for the Openflamingo vision-language
model, with retrieval set to 1, 3, 5, 7 and 13 images.
Table 3 shows that when a single image is retrieved
(N=1), both configurations give random results with a mAcc
of 50%. However, as the number of images retrieved
increases, the benefits of fine-tuning become clear. For
instance, with N=3, RA VID with the pre-trained CLIP
achieves a mAcc of 79.79%, while with the fine-tuned CLIP,
it reaches 93.53%, an absolute improvement of around
13.7%. Similar gains occur for N=5, 7, and 13, where
RA VID with fine-tuned CLIP consistently outperforms the
pre-trained version by around 12 to 12.5% in average ac-
curacy. Improvements are not only evident at average ac-
curacy but are also significant across multiple generative
models. With N=3, notable gains can be seen in the indi-
vidual model subsets: for example, StyleGAN accuracy in-
creased from 76.91% →95.94%, while BigGAN improved
from 86.93% →99.25%. The effect is also striking in the
DeepFakes subset, where accuracy jumped from 70.53%
→92.64%, highlighting the ability of fine-tuned embed-
dings to enhance model generalization. This trend remains
consistent across other retrieval settings (N=5, 7, and 13),
where fine-tuned CLIP embeddings yield substantial im-
provements across nearly all generative models. Figure 8
shows the impact of fine-tuning the CLIP image encoder.
The W/ RAG RA VID (*) bars (representing RA VID with
fine-tuned CLIP) consistently exhibit better accuracy than
the W/ RAG RA VID bars (RA VID with pre-trained CLIP)
for N=3 and N=13, suggesting that fine-tuning improves
the retrieval of relevant context, resulting in better detection
performance.
These findings suggest that pre-trained CLIP lacks sen-
sitivity to AI-generation artifacts, as it has been originally
trained for generic vision-language tasks rather than foren-
sic detection. Fine-tuning aligns the embedding space with
9

/uni00000020/uni00000014
 /uni00000020/uni00000015
 /uni00000020/uni00000016
 /uni00000024/uni00000059/uni0000004a
/uni00000025/uni0000004f/uni00000058/uni00000055/uni00000003/uni0000002f/uni00000048/uni00000059/uni00000048/uni0000004f/uni00000003/uni0000000b /uni0000000c
/uni00000013/uni00000015/uni00000013/uni00000017/uni00000013/uni00000019/uni00000013/uni0000001b/uni00000013/uni00000030/uni00000048/uni00000044/uni00000051/uni00000003/uni00000024/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000008/uni0000000c/uni00000018/uni00000018/uni00000011/uni00000016/uni0000001a
/uni00000017/uni0000001c/uni00000011/uni0000001a/uni00000017/uni00000018/uni00000018/uni00000011/uni00000019/uni00000016/uni0000001b/uni0000001c/uni00000011/uni00000018/uni0000001c
/uni0000001b/uni00000014/uni00000011/uni0000001b/uni00000014
/uni0000001a/uni00000019/uni00000011/uni00000017/uni00000019
/uni00000018/uni00000016/uni00000011/uni00000018/uni0000001b/uni0000001b/uni00000015/uni00000011/uni00000019/uni00000015/uni00000025/uni0000004f/uni00000058/uni00000055/uni00000003/uni00000026/uni00000052/uni00000050/uni00000053/uni00000044/uni00000055/uni0000004c/uni00000056/uni00000052/uni00000051
/uni0000001b/uni00000013 /uni0000001a/uni00000013 /uni00000019/uni00000013 /uni00000018/uni00000013 /uni00000017/uni00000013 /uni00000024/uni00000059/uni0000004a
/uni0000002d/uni00000033/uni00000028/uni0000002a/uni00000003/uni00000034/uni00000058/uni00000044/uni0000004f/uni0000004c/uni00000057/uni0000005c/uni00000019/uni0000001c/uni00000011/uni00000016/uni00000015 /uni0000001a/uni00000013/uni00000011/uni00000018/uni00000017/uni00000019/uni0000001b/uni00000011/uni0000001a/uni00000013 /uni00000019/uni0000001b/uni00000011/uni00000014/uni00000019/uni0000001a/uni00000013/uni00000011/uni00000013/uni00000015/uni0000001b/uni00000016/uni00000011/uni00000017/uni00000019/uni0000001a/uni0000001c/uni00000011/uni0000001b/uni0000001a /uni0000001a/uni0000001b/uni00000011/uni0000001a/uni00000016 /uni0000001a/uni0000001a/uni00000011/uni00000016/uni00000016/uni0000001a/uni00000017/uni00000011/uni0000001c/uni00000016
/uni00000019/uni0000001c/uni00000011/uni00000016/uni00000018/uni0000001a/uni0000001b/uni00000011/uni0000001b/uni00000019/uni0000002d/uni00000033/uni00000028/uni0000002a/uni00000003/uni00000026/uni00000052/uni00000050/uni00000053/uni00000055/uni00000048/uni00000056/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000026/uni00000052/uni00000050/uni00000053/uni00000044/uni00000055/uni0000004c/uni00000056/uni00000052/uni00000051/uni00000026/uni00000015/uni00000033/uni00000010/uni00000026/uni0000002f/uni0000002c/uni00000033 /uni00000035/uni00000024/uni00000039/uni0000002c/uni00000027/uni00000003/uni00000032/uni00000053/uni00000048/uni00000051/uni00000049/uni0000004f/uni00000044/uni00000050/uni0000004c/uni00000051/uni0000004a/uni00000052 /uni00000024/uni00000059/uni0000004a/uni00000003/uni00000026/uni00000015/uni00000033/uni00000010/uni00000026/uni0000002f/uni0000002c/uni00000033 /uni00000024/uni00000059/uni0000004a/uni00000003/uni00000035/uni00000024/uni00000039/uni0000002c/uni00000027/uni00000003/uni00000032/uni00000053/uni00000048/uni00000051/uni00000049/uni0000004f/uni00000044/uni00000050/uni0000004c/uni00000051/uni0000004a/uni00000052Figure 7. RA VID’s robustness under gaussian blur and JPEG compression, common real-world degradations affecting AI-generated image
detection.
Table 4. Performance (ACC) after applying common image degradations.
Methods VLMs DegradationGAN Deep
FakesLow level Perceptual lossGuidedLDM GlideDalle mAcc
Pro-
GANCycle-
GANBig-
GANStyle-
GANGau-
GANStar-
GANSITD SAN CRN IMLE200
steps200
w/cfg100
steps100
2750
27100
10
C2P-CLIP - Blur σ= 1 96.10 90.31 97.02 97.00 95.75 96.80 93.43 95.56 57.08 68.84 68.84 47.90 01.20 06.60 01.60 12.50 13.30 10.60 01.60 55.37
C2P-CLIP - Blur σ= 2 72.20 85.24 87.35 79.45 90.08 86.47 80.33 95.56 51.60 60.90 61.17 45.40 01.20 06.60 01.60 12.50 13.30 10.60 03.50 49.74
C2P-CLIP - Blur σ= 3 77.20 84.18 77.03 62.87 87.19 88.64 75.52 95.00 49.54 56.39 56.39 47.80 13.50 35.30 12.20 42.00 40.40 42.20 13.70 55.63
RA VID (N=13) Openflamingo Blurσ= 1 96.50 90.46 97.28 96.90 96.17 97.47 93.23 95.00 56.85 74.12 74.11 73.35 97.40 94.30 97.25 91.25 90.90 92.75 97.00 89.59
RA VID (N=13) Openflamingo Blurσ= 2 73.73 85.47 87.83 79.99 91.12 88.44 81.35 95.00 51.37 65.37 66.07 73.40 94.05 88.65 94.30 80.80 82.05 81.10 94.30 81.81
RA VID (N=13) Openflamingo Blurσ= 3 78.51 84.56 76.78 63.35 88.10 90.17 76.47 95.00 49.54 59.97 60.15 70.30 87.80 75.70 88.25 73.30 74.25 73.20 87.25 76.46
C2P-CLIP - Jpeq q= 80 95.80 94.93 92.92 75.60 96.85 95.02 85.57 94.72 55.94 92.53 92.42 64.80 14.30 53.10 16.80 58.40 63.40 56.80 17.10 69.32
C2P-CLIP - Jpeq q= 70 94.49 94.44 87.10 65.30 95.18 92.72 84.90 93.89 54.79 86.45 88.09 70.30 24.10 67.70 27.90 61.70 64.10 56.70 30.50 70.54
C2P-CLIP - Jpeq q= 60 94.59 94.40 81.80 62.30 95.57 89.62 82.76 90.56 53.65 80.00 80.05 65.90 23.20 69.70 26.10 58.00 60.50 54.50 42.10 68.70
C2P-CLIP - Jpeq q= 50 93.79 93.26 80.55 60.17 94.69 92.12 78.74 88.06 52.97 76.64 73.89 63.40 25.70 71.40 25.20 59.90 62.60 56.50 45.40 68.16
C2P-CLIP - Jpeq q= 40 93.23 91.79 75.55 57.20 93.22 92.62 75.10 81.39 52.74 77.96 75.56 71.50 35.10 77.80 34.60 64.90 65.10 62.20 52.80 70.02
RA VID (N=13) Openflamingo Jpeqq= 80 95.90 95.31 92.68 75.07 97.00 96.00 84.14 93.61 55.71 92.96 93.32 66.15 91.20 70.95 89.90 69.20 66.65 70.15 89.80 83.46
RA VID (N=13) Openflamingo Jpeqq= 70 94.24 94.55 86.58 65.05 95.09 93.85 84.16 93.33 54.57 83.85 86.38 63.65 86.20 64.60 84.25 68.00 66.50 70.20 82.55 79.87
RA VID (N=13) Openflamingo Jpeqq= 60 94.49 94.47 81.23 62.09 95.42 91.50 82.26 89.72 53.20 82.07 82.23 65.55 86.85 63.35 85.45 69.45 68.20 71.30 77.10 78.73
RA VID (N=13) Openflamingo Jpeqq= 50 93.83 93.30 80.05 60.07 94.58 93.72 77.91 87.22 52.28 78.78 76.04 66.80 85.55 62.10 85.70 68.75 67.60 70.15 74.90 77.33
RA VID (N=13) Openflamingo Jpeqq= 40 93.01 91.98 74.98 57.05 92.99 92.90 74.01 80.83 52.51 77.20 75.08 63.70 81.65 60.40 81.60 66.90 66.50 68.10 72.30 74.93
C2P-CLIP - Average 89.68 91.07 84.91 69.99 93.57 91.75 82.04 91.84 53.54 74.96 74.55 59.62 17.29 48.52 18.25 46.24 47.84 43.76 25.84 63.44
RA VID (N=13) Openflamingo Average 90.03 91.26 84.68 69.95 93.81 93.01 81.69 91.21 53.25 76.79 76.67 67.86 88.84 72.51 88.34 73.46 72.83 74.62 84.4 80.27
Table 5. Performance comparison of different Vision-Language Models (VLMs) on RA VID’s detection task.
VLMs NGAN Deep
FakesLow level Perceptual lossGuidedLDM GlideDalle mAcc
Pro-
GANCycle-
GANBig-
GANStyle-
GANGau-
GANStar-
GANSITD SAN CRN IMLE200
steps200
w/cfg100
steps100
2750
27100
10
Gemma3 177.84 67.68 61.12 57.39 77.17 70.79 55.71 59.17 61.87 50.10 50.09 49.20 70.95 61.80 70.65 69.95 69.80 71.00 60.70 63.84
Gemma3 392.46 89.82 76.88 82.34 90.81 70.94 59.93 71.94 63.24 56.83 56.86 58.90 86.20 78.10 85.35 83.55 83.10 84.40 79.85 76.19
Gemma3 596.03 91.56 86.30 88.60 94.54 83.62 68.81 72.50 64.84 63.61 63.63 64.15 91.50 86.95 91.55 88.65 89.35 89.90 88.60 82.35
Gemma3 797.10 93.79 92.73 91.06 96.59 89.64 69.95 75.83 64.38 80.74 80.77 65.70 94.00 91.10 93.95 91.00 91.20 92.05 92.20 86.52
Gemma3 13 97.34 92.73 92.92 89.68 95.55 93.42 76.11 72.22 62.33 88.62 88.37 67.25 95.60 92.45 95.55 92.50 92.90 93.30 93.60 88.02
Qwen-VL 191.29 93.60 83.60 66.23 80.19 93.70 90.84 88.89 59.36 88.89 94.58 63.01 94.89 88.39 94.89 89.09 89.44 90.84 90.69 85.92
Qwen-VL 399.49 97.27 94.77 93.16 96.11 99.77 92.45 93.33 62.33 97.91 98.00 66.60 99.00 95.55 99.10 92.90 93.05 94.55 97.75 92.79
Qwen-VL 599.92 97.80 98.10 95.84 99.12 99.85 93.01 93.61 62.79 97.01 97.03 67.60 99.30 96.40 99.30 93.70 94.10 95.00 98.25 93.56
Qwen-VL 799.69 96.29 95.95 94.02 97.57 99.85 93.10 92.22 61.87 97.30 97.32 66.95 99.25 95.60 99.20 93.45 93.85 94.70 97.55 92.93
Qwen-VL 13 99.96 97.84 98.70 95.24 99.28 99.82 93.36 93.61 63.01 96.47 96.46 67.80 99.30 96.75 99.40 94.05 95.00 95.70 98.40 93.69
Openflamingo 150.00 50.00 50.00 50.00 50.00 50.00 50.08 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00
Openflamingo 399.95 97.84 99.25 95.94 99.38 99.85 92.64 93.61 62.33 97.55 97.59 66.95 99.35 96.35 99.35 93.10 93.25 94.40 98.40 93.53
Openflamingo 599.95 97.58 99.28 96.24 99.30 99.82 93.19 93.89 63.24 96.26 96.25 68.10 99.20 96.45 99.30 94.00 94.35 95.10 98.30 93.67
Openflamingo 799.96 97.46 99.15 96.24 99.32 99.80 93.30 94.72 63.93 96.04 96.03 68.30 99.25 96.65 99.35 94.00 94.70 95.30 98.15 93.77
Openflamingo 13 99.98 97.35 99.15 96.27 99.33 99.82 93.47 95.00 63.70 95.53 95.56 68.75 99.20 96.95 99.35 94.65 94.90 95.85 98.35 93.85
AI-generated image distributions, improving retrieval effec-
tiveness and enabling the model to extract subtle features
that align with the context of AI-generated image detection.
This analysis highlights the critical role of domain-specific
fine-tuning in retrieval-augmented AI-generated image de-tection. While trained CLIP captures general visual seman-
tics, it fails to retrieve the most relevant images for foren-
sic analysis. In contrast, fine-tuned RA VID CLIP learns
generative-aware representations, ensuring that retrieved
images provide meaningful context, ultimately leading to
10

state-of-the-art detection performance.
4.6. Robustness to Image Degradation
To systematically evaluate the robustness of our proposed
approach, we assess its performance under two common
forms of image degradation: Gaussian blur and JPEG com-
pression. These perturbations simulate real-world chal-
lenges where images undergo quality loss due to compres-
sion artifacts or motion blur, which can adversely impact
AI-generated image detection. We compare our method,
RA VID (N=13) W/ RAG Openflamingo, against the base-
line C2P-CLIP, analyzing their degradation trends across
multiple generative models. The quantitative results are
summarized in Table 4, and the performance trends under
different blur and JPEG compression levels are illustrated
in Figure 7.
4.6.1. Gaussian Blur Degradation
We apply Gaussian blur with increasing standard deviations
(σ= 1,2,3) to evaluate the resilience of both methods un-
der varying levels of spatial smoothing. As expected, per-
formance degrades as blur severity increases, but RA VID
exhibits significantly higher robustness. At σ= 1, RA VID
achieves a mean accuracy (mAcc) of 89.59%, substantially
outperforming C2P-CLIP (55.37%). Notably, under se-
vere degradation σ= 3, RA VID maintains an accuracy of
76.46%, whereas C2P-CLIP deteriorates to 55.63%. These
results highlight the effectiveness of retrieval-augmented
generation in preserving discriminative features even under
heavy spatial distortions.
4.6.2. JPEG Compression Robustness
To simulate real-world compression artifacts, we apply
JPEG degradation with quality factors ranging from q=80
(minimal compression) to q=40 (high compression). Both
methods experience a decline in accuracy as compression
severity increases, shown in Figure 7. However, RA VID
consistently outperforms C2P-CLIP across all quality lev-
els. At q=80, RA VID achieves 83.46%, compared to
69.32% for C2P-CLIP. Even at the lowest quality setting
(q=40), RA VID maintains a 74.93% accuracy, outperform-
ing C2P-CLIP (70.02%) despite the loss of high-frequency
details. The performance gap further underscores the ad-
vantages of retrieval-augmented visual-language models in
handling lossy compression artifacts.
4.6.3. Discussion
Across both degradation types, RA VID consistently demon-
strates greater robustness compared to the baseline. This
can be attributed to its retrieval-augmented mechanism,
which enhances contextual understanding by leveraging ex-
ternal image priors. By integrating semantically relevant
information, RA VID is able to recover essential details thatare lost due to degradation, improving the overall detec-
tion accuracy. Notably, RA VID also outperforms tradi-
tional methods in robustness, maintaining high accuracy
even under image degradations such as Gaussian blur and
JPEG compression. Specifically, RA VID achieves an aver-
age accuracy of 80.27% under degradation conditions, com-
pared to 63.44% for the state-of-the-art model C2P-CLIP,
demonstrating consistent improvements in Gaussian blur
and JPEG compression scenarios. These findings reinforce
the effectiveness of retrieval-augmented visual models for
real-world scenarios, where image quality is often compro-
mised due to pre-processing pipelines, social media com-
pression, or camera limitations.
5. Impact of Vision-Language Model Selection
To evaluate the influence of different vision-language mod-
els (VLMs) on the detection ability of RA VID, we con-
ducted experiments using Qwen-VL, OpenFlamingo, and
Gemma while varying the retrieval count (N). Table 5
presents the results, highlighting the varying effectiveness
of each VLM. The findings provide key insights into each
model’s generalization ability, robustness to different gen-
erative techniques, and sensitivity to the amount of context.
Performance Across Generative Models
Across generative categories, including multiple GAN vari-
ants, perceptual loss-based approaches, deepfakes, and ad-
vanced text-image diffusion models, VLMs have distinct
strengths and limitations.
Qwen-VL consistently performs competently with its
counterpart, Openflamingo, across almost all generative
models. Its high detection rates, particularly in Pro-GAN,
StarGAN, and diffusion-based models (e.g., LDM, Glide,
and DALL·E), suggest strong multimodal understanding
and generalization. Even in more challenging settings such
as IMLE, and CRN, Qwen-VL maintains impressive accu-
racy, typically above 96%. This robustness indicates that
Qwen-VL can effectively leverage both visual and linguis-
tic cues for fine-grained detection tasks, regardless of the
manipulation’s generation mechanism.
OpenFlamingo , while exhibiting poor one-shot perfor-
mance, a 50% mAcc, indicative of random prediction,
rapidly improves with higher shot counts. From three shots,
it matches or slightly trails Qwen-VL in several categories.
For example, at N= 13 , OpenFlamingo reaches a competi-
tive 93.85% mAcc, closely higher than Qwen-VL’s 93.69%.
Its trajectory suggests a firm reliance on contextual support,
potentially making it well-suited to few-shot or retrieval-
augmented learning environments but less reliable in zero-
or one-shot scenarios.
Gemma3 , in contrast, exhibits a more gradual improve-
ment as the number of shots increases. From N= 1, we
observe ongoing gains reaching 88.02% at N= 13 . While
11

/uni00000053/uni00000055/uni00000052/uni0000004a/uni00000044/uni00000051/uni00000046/uni0000005c/uni00000046/uni0000004f/uni00000048/uni0000004a/uni00000044/uni00000051/uni00000045/uni0000004c/uni0000004a/uni0000004a/uni00000044/uni00000051/uni00000056/uni00000057/uni0000005c/uni0000004f/uni00000048/uni0000004a/uni00000044/uni00000051/uni0000004a/uni00000044/uni00000058/uni0000004a/uni00000044/uni00000051 /uni00000056/uni00000057/uni00000044/uni00000055/uni0000004a/uni00000044/uni00000051/uni00000047/uni00000048/uni00000048/uni00000053/uni00000049/uni00000044/uni0000004e/uni00000048/uni00000056/uni00000048/uni00000048/uni0000004c/uni00000051/uni0000004a/uni00000047/uni00000044/uni00000055/uni0000004e/uni00000056/uni00000044/uni00000051 /uni00000046/uni00000055/uni00000051/uni0000004c/uni00000050/uni0000004f/uni00000048
/uni0000004a/uni00000058/uni0000004c/uni00000047/uni00000048/uni00000047/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000015/uni00000013/uni00000013
/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000015/uni00000013/uni00000013/uni00000042/uni00000046/uni00000049/uni0000004a/uni0000004f/uni00000047/uni00000050/uni00000042/uni00000014/uni00000013/uni00000013
/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000018/uni00000013/uni00000042/uni00000015/uni0000001a/uni0000004a/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000042/uni00000014/uni00000013/uni00000013/uni00000042/uni00000014/uni00000013/uni00000047/uni00000044/uni0000004f/uni0000004f/uni00000048/uni00000018/uni00000013/uni00000019/uni00000013/uni0000001a/uni00000013/uni0000001b/uni00000013/uni0000001c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000024/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000008/uni0000000c/uni0000003a/uni00000012/uni00000003/uni00000035/uni00000024/uni0000002a/uni00000003/uni0000000b/uni0000000d/uni0000000c/uni0000000f/uni00000003/uni00000031/uni00000020/uni00000016 /uni0000003a/uni00000012/uni00000003/uni00000035/uni00000024/uni0000002a/uni00000003/uni0000000b/uni0000000d/uni0000000c/uni0000000f/uni00000003/uni00000031/uni00000020/uni00000014/uni00000016 /uni0000003a/uni00000012/uni00000003/uni00000035/uni00000024/uni0000002a/uni0000000f/uni00000003/uni00000031/uni00000020/uni00000016 /uni0000003a/uni00000012/uni00000003/uni00000035/uni00000024/uni0000002a/uni0000000f/uni00000003/uni00000031/uni00000020/uni00000014/uni00000016Figure 8. Evaluation of CLIP Task-Tuning’s Impact on Contextual Image Retrieval. We compare image retrieval quality between the
original pretrained CLIP and a fine-tuned variant (RA VID CLIP), evaluating how task-specific tuning affects the relevance of retrieved
images in RA VID detection capacity.
Table 6. Generalization performance of methods trained on 4-class ProGAN. Results show accuracy (%) on real and synthetic data subsets,
each containing 3,000 image samples.
Methods #params MS COCO Flickr ControlNet Dalle3 DiffusionDB IF LaMA LTE SD2Inpaint SDXL SGXL SD3 mAcc
FatFormer 493M 33.97 34.04 28.27 32.07 28.10 27.95 28.67 12.37 22.63 31.97 22.23 35.91 28.18
RINE 434M 99.80 99.90 91.60 75.00 73.00 77.40 30.90 98.20 71.90 22.20 98.50 08.30 70.56
C2P-CLIP 304M 99.67 99.73 15.10 75.57 27.87 89.56 65.43 00.20 27.90 82.90 07.17 70.46 55.13
RA VID (N=13) - 97.83 99.23 85.80 68.93 70.70 60.71 62.97 99.97 80.37 62.10 98.80 58.31 78.81
N=1 N=3 N=5 N=7 N=13
#images020406080Average accuracy (%)63.8476.1982.3586.5288.02
85.9292.79 93.56 92.7993.69
50.0093.53 93.67 93.77 93.85Gemma3 Qwen-VL Openflamingo
Figure 9. Impact of Vision-Language Model Selection on RA VID
Performance. We assess how the vision-language models (VLMs)
influence RA VID’s detection capabilities by evaluating three mod-
els under varying retrieval counts (N): Qwen-VL, OpenFlamingo,
and Gemma3.
Gemma3 performs solidly across GAN-based generations
and maintains decent results on LDM and Glide models, it
shows noticeable weaknesses in perceptual loss and lower-level models. This pattern suggests Gemma3 may have less
sensitivity to nuanced texture or distributional artifacts that
other models pick up on more easily.
Sensitivity to Number of Shot
Figure 9 highlights the progression of average accuracy
(mAcc) of VLMs with increasing retrieval count N. The
trends are revealing:
• Qwen-VL shows high performance from the outset, start-
ing at 85.92% with just one shot and reaching near-
maximum capacity, 92.79%, after three shots. This sug-
gests the model is inherently robust and benefits modestly
from additional contexts.
• OpenFlamingo shows a 50% jump to over 93% mAcc
between one and three shots. From there, performance
stagnates, with minor gains up to 93.85% at 13 shots.
This pattern reflects a steep learning curve, highlighting
its adaptability in few-shot contexts.
• Gemma3 improves more linearly, its mAcc rising consis-
tently from 63.84% at N= 1 to 88.02% at N= 13 .
Unlike OpenFlamingo, it shows a slower rate of improve-
ment but a consistent trajectory, which might favor use
cases where the retrieval count is expected to scale over
12

time.
These trends indicate a practical trade-off between initial
generalization capacity and few-shot adaptability. Qwen-
VL is a strong candidate for out-of-the-box performance
or low-shot applications, while OpenFlamingo is prefer-
able in scenarios with a richer support context available.
Gemma3’s progressive improvement makes it suitable for
pipelines where inference is augmented incrementally.
Implications for RA VID
These results underscore the value of advanced vision-
language alignment in improving AI-generated image de-
tection. Qwen-VL’s superior performance suggests that
high-capacity models with strong vision-language fusion
mechanisms can reliably detect images from various gen-
erative models. Furthermore, the stark contrast between
OpenFlamingo’s one-shot and few-shot performance re-
veals the importance of retrieval augmentation in practi-
cal deployments of VLMs on AI-generated image detection
tasks. In practical settings, the choice of model and retrieval
strategy should be guided by task constraints: Qwen-VL
excels in a few context situations, OpenFlamingo benefits
from enriched support, and Gemma3 provides a consistent
baseline.
6. Generalization on Unseen Data
To assess cross-domain robustness, we evaluate the gener-
alization performance of four top-performing methods from
Table 1, FatFormer, RINE, C2P-CLIP, and RA VID. Each
model is trained solely on the ProGAN 4-class dataset and
tested on a broad spectrum of unseen data sources, in-
cluding authentic images (e.g., MS COCO, Flickr) and di-
verse generative models (e.g., ControlNet, DALL·E 3, Dif-
fusionDB, SGXL, LTE, etc...). This evaluation simulates a
realistic open-world scenario where detection models face
data distributions that deviate significantly from their train-
ing domain.
As shown in Table 6, RA VID exhibits strong cross-
domain generalization, achieving the highest overall mean
accuracy of 78.81% across both real and synthetic do-
mains. While other methods demonstrate strengths in spe-
cific datasets, they often suffer from instability under dis-
tribution shifts. RINE, for instance, performs exceptionally
well on MS COCO and Flickr but collapses on SD3, re-
flecting limited robustness to unseen generative processes.
C2P-CLIP attains high accuracy on IF and DALL·E 3 but
fails drastically on LTE, revealing poor transferability un-
der localized editing. In contrast, RA VID maintains con-
sistent and competitive performance across nearly all test
domains, particularly excelling on challenging sources like
LTE (99.97%), SGXL (98.80%), and SD2Inpaint (80.37%).
Despite being trained on a single synthetic domain, itsstrong performance on real and synthetic datasets under-
scores its capacity to learn transferable representations,
making it a compelling candidate for reliable deployment
in open-world scenarios.
7. Conclusion
In this paper, we have introduced RA VID, a novel retrieval-
augmented framework for detecting AI-generated images.
By dynamically retrieving and integrating relevant visual
knowledge, RA VID enhances detection accuracy and gen-
eralization. Unlike traditional methods that rely on low-
level artifacts or model fingerprints, our approach leverages
a fine-tuned CLIP-based image encoder (RA VID CLIP) for
embedding generation and retrieval. Additionally, we incor-
porate VLMs like Openflamingo to enrich contextual un-
derstanding. Retrieving semantically relevant images from
a vector database and integrating them into the detection
pipeline significantly improves performance. Evaluations
on the UniversalFakeDetect benchmark (spanning 19 gen-
erative models) showed that RA VID outperforms existing
methods, achieving 93.85% accuracy in both in- and out-of-
domain settings. A detailed analysis revealed a 35.51% per-
formance gap between setups W/ and W/O retrieval in the
3-shot setting, highlighting the critical role of relevant con-
text. Our findings confirm that retrieval not only enhances
accuracy but also scales with additional retrieval shots, re-
inforcing its impact. Furthermore, our robustness analy-
sis demonstrated that RA VID maintains superior detection
performance under commun image degradations, including
Gaussian blur and JPEG compression. In contrast baseline
methods struggle with spatial distortions and compression
artifacts, RA VID leverages retrieval-augmented generation
to incorporate relevant context, maintaining robust perfor-
mance and enabling significantly higher accuracy across
all tested degradation levels. It achieves an average accu-
racy of 80.27% under degradation conditions, compared to
63.44% for the state-of-the-art model C2P-CLIP, demon-
strating consistent improvements in both Gaussian blur and
JPEG compression scenarios. These results highlight the
resilience of retrieval-based approaches in real-world con-
ditions where images often suffer from quality loss due to
compression pipelines, motion blur, or other distortions. As
generative models rapidly evolve, retrieval-augmented tech-
niques like RA VID will be essential for developing more
robust, adaptable AI-generated content detection systems.
Beyond improving detection accuracy, RA VID paves the
way for advancements in context-aware and resilient detec-
tion mechanisms.
Acknowledgments
This work has been partially funded by the project
PCI2022-134990-2 (MARTINI) of the CHISTERA IV Co-
13

fund 2021 program. Abdenour Hadid is funded by Total-
Energies collaboration agreement with Sorbonne University
Abu Dhabi.
References
[1] Adobe. Create with firefly generative ai. https://www.
adobe.com/products/firefly.html , 2023. Ac-
cessed: 2024-10-10. 3
[2] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf
Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton,
Samir Gadre, Shiori Sagawa, et al. Openflamingo: An open-
source framework for training large autoregressive vision-
language models. arXiv preprint arXiv:2308.01390 , 2023.
2, 7
[3] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large
scale gan training for high fidelity natural image synthesis. In
International Conference on Learning Representations . 3, 6
[4] Lucy Chai, David Bau, Ser-Nam Lim, and Phillip Isola.
What makes fake images detectable? understanding proper-
ties that generalize. In Computer Vision–ECCV 2020: 16th
European Conference, Glasgow, UK, August 23–28, 2020,
Proceedings, Part XXVI 16 , pages 103–120. Springer, 2020.
7
[5] You-Ming Chang, Chen Yeh, Wei-Chen Chiu, and Ning Yu.
Antifakeprompt: Prompt-tuned vision-language models are
fake image detectors. CoRR , 2023. 3, 6
[6] Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun.
Learning to see in the dark. In Proceedings of the IEEE con-
ference on computer vision and pattern recognition , pages
3291–3300, 2018. 6
[7] Qifeng Chen and Vladlen Koltun. Photographic image syn-
thesis with cascaded refinement networks. In Proceedings of
the IEEE international conference on computer vision , pages
1511–1520, 2017. 6
[8] Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha,
Sunghun Kim, and Jaegul Choo. Stargan: Unified genera-
tive adversarial networks for multi-domain image-to-image
translation. In Proceedings of the IEEE conference on
computer vision and pattern recognition , pages 8789–8797,
2018. 2, 3, 6
[9] Davide Cozzolino, Giovanni Poggi, Riccardo Corvi,
Matthias Nießner, and Luisa Verdoliva. Raising the bar of
ai-generated image detection with clip. In 2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
Workshops (CVPRW) , pages 4356–4366. IEEE, 2024. 2
[10] Tao Dai, Jianrui Cai, Yongbing Zhang, Shu-Tao Xia, and
Lei Zhang. Second-order attention network for single im-
age super-resolution. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition , pages
11065–11074, 2019. 6
[11] Prafulla Dhariwal and Alexander Nichol. Diffusion models
beat gans on image synthesis. Advances in neural informa-
tion processing systems , 34:8780–8794, 2021. 2, 3, 6
[12] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing
Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and
Yoshua Bengio. Generative adversarial nets. Advances in
neural information processing systems , 27, 2014. 3[13] Zhengchao Huang, Bin Xia, Zicheng Lin, Zhun Mou, and
Wenming Yang. Ffaa: Multimodal large language model
based explainable open-world face forgery analysis assistant.
arXiv preprint arXiv:2408.10072 , 2024. 3
[14] Sofia Iliopoulou, Panagiotis Tsinganos, Dimitris Ampeliotis,
and Athanassios Skodras. Synthetic face discrimination via
learned image compression. Algorithms , 17(9):375, 2024. 3
[15] Soyeong Jeong, Kangsan Kim, Jinheon Baek, and Sung Ju
Hwang. Videorag: Retrieval-augmented generation over
video corpus. arXiv preprint arXiv:2501.05874 , 2025. 3
[16] Yonghyun Jeong, Doyeon Kim, Seungjai Min, Seongho Joe,
Youngjune Gwon, and Jongwon Choi. Bihpf: Bilateral high-
pass filters for robust deepfake detection. In Proceedings of
the IEEE/CVF Winter Conference on Applications of Com-
puter Vision , pages 48–57, 2022. 3
[17] Yonghyun Jeong, Doyeon Kim, Youngmin Ro, and Jongwon
Choi. Frepgan: robust deepfake detection using frequency-
level perturbations. In Proceedings of the AAAI conference
on artificial intelligence , pages 1060–1068, 2022.
[18] Yonghyun Jeong, Doyeon Kim, Youngmin Ro, Pyounggeon
Kim, and Jongwon Choi. Fingerprintnet: Synthesized fin-
gerprints for generated image detection. In European Con-
ference on Computer Vision , pages 76–94. Springer, 2022.
3
[19] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen.
Progressive growing of gans for improved quality, stability,
and variation. In International Conference on Learning Rep-
resentations , 2018. 2, 3, 6
[20] Tero Karras, Samuli Laine, and Timo Aila. A style-based
generator architecture for generative adversarial networks.
InProceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition , pages 4401–4410, 2019. 2, 3,
6
[21] Mamadou Keita, Wassim Hamidouche, Hassen Bougueffa,
Abdenour Hadid, and Abdelmalik Taleb-Ahmed. Harness-
ing the power of large vision language models for synthetic
image detection. arXiv preprint arXiv:2404.02726 , 2024. 6
[22] Mamadou Keita, Wassim Hamidouche, Hessen Bougu-
effa Eutamene, Abdelmalik Taleb-Ahmed, David Camacho,
and Abdenour Hadid. Bi-lora: A vision-language approach
for synthetic image detection. Expert Systems , 42(2):e13829,
2025. 3, 6
[23] Christos Koutlis and Symeon Papadopoulos. Leveraging rep-
resentations from intermediate encoder-blocks for synthetic
image detection. In European Conference on Computer Vi-
sion, pages 394–411. Springer, 2024. 3, 7
[24] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp
tasks. Advances in neural information processing systems ,
33:9459–9474, 2020. 2
[25] Ke Li, Tianhao Zhang, and Jitendra Malik. Diverse image
synthesis from semantic layouts via conditional imle. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision , pages 4220–4229, 2019. 6
14

[26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning. Advances in neural information
processing systems , 36:34892–34916, 2023. 3
[27] Huan Liu, Zichang Tan, Chuangchuang Tan, Yunchao Wei,
Jingdong Wang, and Yao Zhao. Forgery-aware adaptive
transformer for generalizable synthetic image detection. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , pages 10770–10780, 2024. 7
[28] MidJourney. Midjourney v5. https : / / www .
midjourney.com , 2023. Accessed: 2024-10-10. 3
[29] Ron Mokady, Amir Hertz, and Amit H Bermano. Clip-
cap: Clip prefix for image captioning. arXiv preprint
arXiv:2111.09734 , 2021. 4
[30] Lakshmanan Nataraj, Tajuddin Manhar Mohammed, Shiv-
kumar Chandrasekaran, Arjuna Flenner, Jawadul H Bappy,
Amit K Roy-Chowdhury, and BS Manjunath. Detecting gan
generated fake images using co-occurrence matrices. arXiv
preprint arXiv:1903.06836 , 2019. 7
[31] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh,
Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya
Sutskever, and Mark Chen. Glide: Towards photorealis-
tic image generation and editing with text-guided diffusion
models. In International Conference on Machine Learning ,
pages 16784–16804. PMLR, 2022. 2, 3, 6
[32] Utkarsh Ojha, Yuheng Li, and Yong Jae Lee. Towards uni-
versal fake image detectors that generalize across genera-
tive models. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages 24480–
24489, 2023. 6, 7
[33] OpenAI. Dall-e 3. https://openai.com/dall-e-3 ,
2023. Accessed: 2024-10-10. 3
[34] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan
Zhu. Gaugan: semantic image synthesis with spatially adap-
tive normalization. In ACM SIGGRAPH 2019 Real-Time
Live! , New York, NY , USA, 2019. Association for Comput-
ing Machinery. 3, 6
[35] Dustin Podell, Zion English, Kyle Lacey, Andreas
Blattmann, Tim Dockhorn, Jonas M ¨uller, Joe Penna, and
Robin Rombach. Sdxl: Improving latent diffusion mod-
els for high-resolution image synthesis. arXiv preprint
arXiv:2307.01952 , 2023. 3
[36] Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing
Shao. Thinking in frequency: Face forgery detection by min-
ing frequency-aware clues. In European conference on com-
puter vision , pages 86–103. Springer, 2020. 7
[37] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray,
Chelsea V oss, Alec Radford, Mark Chen, and Ilya Sutskever.
Zero-shot text-to-image generation. In International confer-
ence on machine learning , pages 8821–8831. Pmlr, 2021. 2,
6
[38] Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei
Yin, and Chao Huang. Videorag: Retrieval-augmented gen-
eration with extreme long-context videos. arXiv preprint
arXiv:2502.01549 , 2025. 3
[39] Monica Riedler and Stefan Langer. Beyond text: Optimizing
rag with multimodal inputs for industrial applications. arXiv
preprint arXiv:2410.21943 , 2024. 3[40] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj ¨orn Ommer. High-resolution image
synthesis with latent diffusion models. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition , pages 10684–10695, 2022. 2, 3, 6
[41] Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Chris-
tian Riess, Justus Thies, and Matthias Nießner. Faceforen-
sics++: Learning to detect manipulated facial images. In
Proceedings of the IEEE/CVF international conference on
computer vision , pages 1–11, 2019. 6
[42] Chitwan Saharia, William Chan, Saurabh Saxena, Lala
Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour,
Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans,
et al. Photorealistic text-to-image diffusion models with deep
language understanding. Advances in neural information
processing systems , 35:36479–36494, 2022. 3
[43] Axel Sauer, Katja Schwarz, and Andreas Geiger. Stylegan-
xl: Scaling stylegan to large diverse datasets. In ACM SIG-
GRAPH 2022 conference proceedings , pages 1–10, 2022. 3
[44] Sergey Sinitsa and Ohad Fried. Deep image fingerprint: To-
wards low budget synthetic image detection and model lin-
eage analysis. In Proceedings of the IEEE/CVF Winter Con-
ference on Applications of Computer Vision , pages 4067–
4076, 2024. 2
[45] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu,
and Yunchao Wei. Learning on gradients: Generalized arti-
facts representation for gan-generated images detection. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition , pages 12105–12114, 2023. 7
[46] Chuangchuang Tan, Renshuai Tao, Huan Liu, Guanghua Gu,
Baoyuan Wu, Yao Zhao, and Yunchao Wei. C2p-clip: Inject-
ing category common prompt in clip to enhance generaliza-
tion in deepfake detection. arXiv preprint arXiv:2408.09647 ,
2024. 3, 4, 5, 6, 7
[47] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu,
Ping Liu, and Yunchao Wei. Frequency-aware deepfake de-
tection: Improving generalizability through frequency space
domain learning. In Proceedings of the AAAI Conference on
Artificial Intelligence , pages 5052–5060, 2024. 7
[48] Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu,
Ping Liu, and Yunchao Wei. Rethinking the up-sampling op-
erations in cnn-based generative network for generalizable
deepfake detection. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition , pages
28130–28139, 2024. 7
[49] Milvus Team. Milvus documentation. https : / /
milvus.io/docs/fr , 2025. Accessed: 2025-03-08. 5
[50] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew
Owens, and Alexei A Efros. Cnn-generated images are
surprisingly easy to spot... for now. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition , pages 8695–8704, 2020. 6, 7
[51] Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun
Wang, Hezhen Hu, Hong Chen, and Houqiang Li. Dire for
diffusion-generated image detection. In Proceedings of the
IEEE/CVF International Conference on Computer Vision ,
pages 22445–22455, 2023. 2, 3, 6
15

[52] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao
Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han,
Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented
generation on multi-modality documents. arXiv preprint
arXiv:2410.10594 , 2024. 3
[53] Xu Zhang, Svebor Karaman, and Shih-Fu Chang. Detecting
and simulating artifacts in gan fake images. In 2019 IEEE in-
ternational workshop on information forensics and security
(WIFS) , pages 1–6. IEEE, 2019. 7
[54] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A
Efros. Unpaired image-to-image translation using cycle-
consistent adversarial networks. In Proceedings of the IEEE
international conference on computer vision , pages 2223–
2232, 2017. 3, 6
16