# Med-SORA: Symptom to Organ Reasoning in Abdomen CT Images

**Authors**: You-Kyoung Na, Yeong-Jun Cho

**Published**: 2025-11-10 06:30:51

**PDF URL**: [https://arxiv.org/pdf/2511.06752v1](https://arxiv.org/pdf/2511.06752v1)

## Abstract
Understanding symptom-image associations is crucial for clinical reasoning. However, existing medical multimodal models often rely on simple one-to-one hard labeling, oversimplifying clinical reality where symptoms relate to multiple organs. In addition, they mainly use single-slice 2D features without incorporating 3D information, limiting their ability to capture full anatomical context. In this study, we propose Med-SORA, a framework for symptom-to-organ reasoning in abdominal CT images. Med-SORA introduces RAG-based dataset construction, soft labeling with learnable organ anchors to capture one-to-many symptom-organ relationships, and a 2D-3D cross-attention architecture to fuse local and global image features. To our knowledge, this is the first work to address symptom-to-organ reasoning in medical multimodal learning. Experimental results show that Med-SORA outperforms existing medical multimodal models and enables accurate 3D clinical reasoning.

## Full Text


<!-- PDF content starts -->

Med-SORA: Symptom to Organ Reasoning in Abdomen CT Images
You-Kyoung Na
Chonnam National University
youkyoung@jnu.ac.krYeong-Jun Cho
Chonnam National University
yj.cho@jnu.ac.kr
Abstract
Understanding symptom-image associations is crucial for
clinical reasoning. However, existing medical multimodal
models often rely on simple one-to-one hard labeling, over-
simplifying clinical reality where symptoms relate to multi-
ple organs. In addition, they mainly use single-slice 2D fea-
tures without incorporating 3D information, limiting their
ability to capture full anatomical context. In this study,
we propose Med-SORA, a framework for symptom-to-organ
reasoning in abdominal CT images. Med-SORA introduces
RAG-based dataset construction, soft labeling with learn-
able organ anchors to capture one-to-many symptom–organ
relationships, and a 2D–3D cross-attention architecture to
fuse local and global image features. To our knowledge,
this is the first work to address symptom-to-organ reason-
ing in medical multimodal learning. Experimental results
show that Med-SORA outperforms existing medical multi-
modal models and enables accurate 3D clinical reasoning.
The code and dataset are available athttps://WILL_
BE_SOON
1. Introduction
Electronic Medical Records (EMRs) are structured clinical
documents that include patient symptoms, diagnoses,
treatments, and other healthcare-related information, as
recorded by healthcare professionals. Understanding the
connection between patient symptoms in EMRs and med-
ical images is an important research goal for advancing our
understanding of clinical decision-making. Such connec-
tions can also serve as supporting evidence for diagnosis,
helping AI systems make more accurate and interpretable
decisions. However, linking symptom descriptions in text
to findings in medical images is technically challenging
due to the gap between abstract language and visual
representations.
Recent studies have explored medical multimodal mod-
els in various applications, including pathology image
analysis [32], medical visual question answering (Med-
VQA) [10], medical report generation [19], and interactive
liverleft kidney
Query:What organ isdisplayed in themarked area?liverVLM ModelMed-SORAQuery:Foamy or bubbly urinereported, suggestingpossible protein leakage.
 liver: 0.57 spleen: 0.13 kidney: 0.52 ✓ High-level     reasoning ✓ one-to-many     matching ✓ Low-level     reasoning ✓ one-to-one     matching(a) Existing approach(b) Our approach
Learnable
¤Figure 1. Comparison between existing medical vision-language
models and the proposed Med-SORA. Ground-truth organs are
marked with⋆.
diagnostic systems [6]. However, existing research has sev-
eral limitations. First, as shown in Fig. 1(a), current medical
multimodal models perform low-level reasoning by identi-
fying organs in marked image areas using one-to-one text-
image correspondence. Such approaches overlook the one-
to-many nature of symptom–organ associations, oversim-
plifying clinical reality. For example, a symptom like ‘jaun-
dice’ can relate to multiple organs (e.g., liver, gallbladder,
pancreas). Second, existing studies [5, 35] rely mainly on
single 2D slice-based features and lack joint 2D-3D analy-
sis, limiting their ability to capture full anatomical context
and spatial relationships.
To address these limitations, we propose the Symptom-
to-Organ Reasoning (Med-SORA), which performs high-
level clinical reasoning by inferring related organs from
patient symptom descriptions as shown in Fig. 1(b). Un-
like existing approaches that rely on direct visual pattern
matching, our method reflects clinical reality where symp-
toms can probabilistically relate to multiple organs. To this
end, we first construct a symptom–text dataset by extracting
symptom–organ knowledge from reliable medical literature
using Retrieval-Augmented Generation (RAG). Second, we
propose soft labeling with learnable organ anchors to model
probabilistic relationships between symptoms and multiple
organs. Third, we fuse 2D and 3D image features from ab-
dominal CT images to capture both fine-grained slice-level
detail and global 3D anatomical structure. Finally, Med-
SORA is trained to align symptom-text and image features
in a shared embedding space.arXiv:2511.06752v1  [cs.CV]  10 Nov 2025

To validate our method, we conducted experiments us-
ing a RAG-based symptom text dataset (validated by med-
ical experts) and the publicly available BTCV abdominal
CT dataset [17]. Results show that soft labeling more effec-
tively captures complex organ relationships in the embed-
ding space than hard labeling. Moreover, our 2D–3D cross-
attention architecture outperformed single-modality 2D and
3D models in organ identification accuracy. Med-SORA
achieved the highest overall performance compared to ex-
isting medical multimodal models. Unlike prior studies that
focus on low-level tasks such as image classification or or-
gan detection, Med-SORA performs high-level clinical rea-
soning. It directly infers and visualizes symptom-related
organs in 3D space, enabling intuitive spatial interpretation.
The main contributions of this work are as follows:
• We propose Med-SORA, a framework that aligns symp-
tom text with 3D CT images for clinical reasoning.
• We build a reliable and clinically valid symptom-organ
dataset using RAG from trusted medical literature.
• We introduce soft labeling with learnable organ anchors
to model one-to-many symptom–organ relationships.
• We design a 2D-3D cross-attention architecture to extract
image features from CT scans for better understanding of
abdominal organs.
• Med-SORA outperforms existing multimodal medical
models on the symptom-to-organ reasoning task.
To the best of our knowledge, Med-SORA is the first at-
tempt to perform symptom-to-organ reasoning in the medi-
cal multimodal domain.
2. Related Works
2.1. RAG-based Construction of Medical Datasets
Retrieval Augmented Generation (RAG) integrates exter-
nal information retrieval to overcome the limitations of
pre-trained language models that rely solely on their in-
ternal knowledge [18]. This approach enables genera-
tive models to search and reference relevant external doc-
uments, thereby enhancing factual accuracy and currency
of the generated content. The medical field especially
benefits from RAG because it can access the latest medi-
cal literature, keeping up with fast-changing research and
treatment guidelines [14]. As a result, RAG-based ap-
proaches [30, 33] have been actively explored in medi-
cal applications, including clinical question answering and
medical knowledge retrieval. Studies [13, 16] have also
employed retrieval-based approaches to automatically con-
struct datasets from large-scale repositories such as PubMed
and Wikipedia.
While these approaches show potential for automated
dataset construction, they primarily focus on generat-
ing data for factual knowledge retrieval and simple
question-answering, making them inadequate for construct-ing datasets requiring complex clinical reasoning.
2.2. Multimodal Learning in the Medical Domain
CLIP [25], a representative vision-language model (VLM),
learns semantic associations between images and text by
training on large-scale image–text pairs. CLIP’s effec-
tiveness in learning generalizable image–text representa-
tions has led to the development of domain-specific vari-
ants for medical applications. For example, PubMed-
CLIP [5] adapts CLIP to the medical domain by retraining
it on PubMed literature and medical images, and BioMed-
CLIP [35] further extends this approach using broader
biomedical data to support diverse medical tasks. These
medical VLMs have been applied to various downstream
tasks, including medical visual question answering [31],
diagnostic imaging assistance [21], and medical educa-
tion [27].
Reasoning across multiple modalities (i.e., text and im-
age) is a key challenge, especially in the medical field where
accurate decision-making depends on integrating various
types of information. In clinical settings, reasoning in-
volves combining symptoms, imaging results, and medical
knowledge to reach a diagnosis. Large language models
like GPT [24] have advanced reasoning capabilities, driv-
ing progress in multimodal reasoning when combined with
image understanding. Recently, text-based reasoning mod-
els have also been developed specifically for the medical
domain. MedMCQA [23] provides a benchmark dataset for
evaluating medical reasoning through multiple-choice clin-
ical questions. Building on such datasets, LLM-based mod-
els like Med-PaLM [28] demonstrate strong performance by
generating answers through reasoning grounded in medical
knowledge. Huy et al. [12] similarly proposed a concept-
based similarity reasoning network for medical image inter-
pretation, which learns region-specific patterns and models
spatial interactions.
However, existing medical reasoning models have sig-
nificant limitations when applied to real clinical scenarios.
Text-based models like Med-PaLM [28] excel at medical
knowledge reasoning but cannot process visual information
from medical images, limiting their applicability in image-
dependent diagnoses. Conversely, medical vision-language
models focus primarily on basic image understanding tasks
rather than complex clinical reasoning that integrates symp-
toms with imaging findings.
3. Proposed Methods
We propose a multimodal framework, Med-SORA, that per-
forms Symptom-to-Organ Reasoning in abdominal CT im-
ages. Med-SORA takes patient symptom text as input and
infers and visualizes related organ locations in 3D abdomi-
nal CT images, as demonstrated in Fig. 1(b). The proposed
Med-SORA consists of three main components: (a) organ-

(b) Soft Labeling for  Symptom Text Embeddings (Sec.3.2)
Knowledge
Source Text
Generation
Retrieval(a) Organ Specific Symptom Text Dataset Construction  (Sec.3.1)
(c) Fusion of 2D and 3D Image Featur es (Sec.3.3)
3D Feature
ExtractorCross
Attention2D Feature
ExtractorSymptom
Text
Linear Projection"Dark brown urine noted...",
"Altered taste sensation...",
"Changes in body odor noted..."
CT Images
...
MLPText Embedding
Anchor Optimization
Soft Label
[0.3, 0.8, 0.2]
FrozenLearnable
Positive AnchorImage FeatureText FeatureLiver
LiverChunk Filtering
"Does this text describe
patient symptoms 
or clinical observations?"Chunk Enhancing
"Expand the symptom
chunks into complete 
sentences"Chunk Enhancing
"Expand the symptom
chunks into complete 
sentences"
Query
"Based ONL Y on given {CHUNK}, create 
200 unique clinical observations ... "Medical Knowledge Base
"Judice characterized by yellowing of skin ...",
...LLMChunking
...
Figure 2. The pipeline of Med-SORA, showing the optimization process foro 1=liver. It consists of (a) RAG-based dataset construction,
(b) soft labeling-based text embedding learning, and (c) 3D-2D feature-based image embedding learning with text-image embedding
alignment.
specific symptom text dataset construction, (b) soft labeling
for symptom text via learnable organ anchors, and (c) fu-
sion of 2D and 3D image features. The overall pipeline of
the proposed method is shown in Fig. 2.
3.1. Organ-Specific Symptom Text Dataset
Construction
To enable multimodal learning between symptom text and
abdominal images, a high-quality text dataset of organ-
specific symptoms is required. However, such datasets are
limited due to the high cost and need for medical expertise.
To address this, we construct a symptom text dataset using
Retrieval-Augmented Generation (RAG) [18], as shown in
Fig. 2(a).
Our symptom text dataset coversNmajor abdominal or-
gans,O={o 1, o2, . . . , o N}, including the liver, kidney,
and stomach. For each organ, symptom texts are collected
from PubMed [1] and Wikipedia [2]. PubMed provides spe-
cialized, clinically validated medical information, whereas
Wikipedia offers more general descriptions. By leveraging
both sources, we aim to capture a wide spectrum of symp-
tom texts. Both platforms support keyword-based retrieval
through their official APIs. Accordingly, we used organ-
specific keywords to extract relevant symptom texts for each
target organ. To standardize sentence length and support ef-
ficient processing, the raw symptom texts are divided into
chunks of two to three sentences.
Some of the retrieved texts after chunking may include
sentences that are not related to symptoms. To improve
quality, a Large Language Model (LLM) is used to ei-
ther rewrite them into clinically relevant symptom descrip-
tions or remove them if they contain irrelevant information
or lack meaningful clinical content. To achieve this, we
prompt the LLM with two instructions: (1)“Does this text
describe a patient’s symptom or clinical observation?”and(2)“Based ONLY on given{CHUNK}, create 200 unique
clinical observations described in complete, medical-style
sentences. ”These prompts guide the LLM to refine and
restructure the texts to better reflect relevant medical infor-
mation. As a result, we generateNtxtsymptom texts,T i=
{ti1, ti2,···, t iNtxt}, for each organo i, without explicitly
mentioning the organ name, enabling inference based solely
on symptom information. The quality and reliability of the
generated symptom text dataset were validated by medical
professionals. The detailed data construction process is de-
scribed in the supplementary materials.
3.2. Soft Labeling for Symptom Text Embeddings
A symptom text is often related to multiple organs rather
than just one. For example, feeling uncomfortable after eat-
ing is intuitively associated with the stomach, but may also
involve other organs, such as the pancreas and gallblad-
der, which contribute to the digestive process. Therefore,
soft labeling is more suitable than hard labeling for learn-
ing symptom–organ embedding spaces. Soft labeling meth-
ods [7, 15] have been proposed, but they rely on a simple
inter-data similarity or predefined threshold. Such methods
are hard to capture the complex and overlapping relation-
ships between symptoms and organs.
To address this limitation, we propose a learnable
anchor-based soft labeling to model complex symp-
tom–organ associations as shown in Fig. 3. Using a pre-
trained text embedding model [20], the symptom textt ijfor
each organ is embedded into vectorsftxt
ij∈Rdtxt(Fig. 3(a)).
iandjdenote the indexes of the organ and the symptom
text, respectively. We then define learnable positive and
negative anchorsv+
i,v−
i∈Rdtxtfor each organo ito cap-
ture complex symptom–organ relationships. As shown in
Fig. 3(b), these anchors are optimized during training:v+
i
learns the distinctive symptom representations specific to

(b) Anchor optimizationpull
pull
pullpush
push
push
(a) Text embedding (c) Soft labeling
Text EmbeddingSoft Label
[0.2, 0.8, 0.3]"Elevated ammonia 
levels documented
in lab tests,  ..."
"Frequent belching
and a sensation of
pressure in the ..."
"Unusual cravings
for sweets reported 
during episodes ..." Positive anchor
Negative anchorText featureFigure 3. Soft labeling for symptom text embeddings via learn-
able anchors. Different colors represent organ classes.↔means
similarity computation, while pull/push operations show the opti-
mization process fors+ands−with marginm.
organo i, whilev−
icaptures general symptom representa-
tions shared across other organs.
The positive and negative similarities between the an-
chors and the symptom embeddingftxt
ijare calculated as fol-
lows:
s+
i(ftxt
ij) =sim(v+
i,ftxt
ij), s−
i(ftxt
kj) =sim(v−
i,ftxt
kj),(1)
wherei̸=k, and the function sim(a,b) =a·b
||a||·||b||repre-
sents cosine similarity. Inspired by triplet loss [9], we define
a margin function as follows:
M 
s+
i, s−
i
= max 
0, m−s+
i
+ max 
0, s−
i−(1−m)
,
(2)
wheremdenotes the margin of the loss. This function en-
forces symptom text embeddings to achieve similaritys+
i
with positive anchors above marginmand similaritys−
i
with negative anchors below(1−m). To train the positive
and negative anchors(v+
i,v−
i)of the organo i, we optimize
the following anchor margin loss.
Lanchor =E ftxt
ij∈P[M(s+
i, s−
i)]+E ftxt
kj∈N[M(s−
i, s+
i)],(3)
wherePdenotes the set of positive symptom vectors associ-
ated with organo i, andNdenotes the set of negative symp-
tom vectors from other organs. Note that these learnable
anchors(v+
i,v−
i)for each organo iare pre-trained in an of-
fline stage using only symptom text data, independently of
the main image–text alignment training.
After training the anchors, we compute soft labels for
each symptom textt kjwith respect to each organo ibased
on the corresponding positive anchorv+
iby
ytxt
oi(tkj) ={sim 
v+
i,ftxt
kj
+ 1}/2.(4)
Considering multiple possibleNorgan associations, the
soft label vector for each symptom textt jis defined as
ytxt(tkj) = [ytxt
o1, ytxt
o2, . . . , ytxt
oN]. Unlike one-hot labels that
assign each symptom to a single organ, our method allows
soft associations with multiple organs as shown in Fig.3(c).
Transformer Encoder
3D Convolution2 0 1CLS32
(b) 3D volume feature extraction(a) 2D image feature extraction
Linear Projection2 0 1CLS16
Cross
Attention
(c) Cross attention3D Patch2D Patch
LearnablePosition+
Patch Embedding
...
Position+
Patch Embedding
MLP
Head
MLP
Head
Transformer Encoder
MLP HeadFigure 4. The proposed 2D-3D feature fusion architecture.
Instead of using softmax, we normalize the cosine similar-
ity score for each soft labelytxt
oito the[0,1]range. This
avoids enforcing a probability distribution and better cap-
tures overlapping organ–symptom relationships.
Based on the obtained soft labelsytxt, we further train
a Multi-Layer Perceptron (MLP) to extract symptom text
features guided byytxt. We define the feature through the
MLP by ¯ftxt=MLP(ftxt). The classification head produces
Noutputs, one for each class, with sigmoid activation ap-
plied to each. The MLP and head are trained by minimiz-
ing a cross-entropy lossL txtbetween the predicted labels
ˆytxt=Head( ¯ftxt)and the soft labelsytxt.
3.3. Fusion of 2D and 3D Image Features
In this section, we propose a feature extraction framework
that jointly captures both 2D and 3D representations from
abdominal CT scans. To enhance organ-level understand-
ing, we extract slice-wise features using a 2D encoder and
volume-level context using a 3D encoder, and integrate
them through a cross-attention mechanism.
An abdominal CT scan consists of multiple 2D axial
slice images, eachl-th image denoted asx l∈RH×W,
whereHandWrepresent the image size. To focus on ma-
jor abdominal organs and normalize the varying number of
CT slices across patients, we sampleDslices for each or-
gan. The resulting input volume of an organ is represented
asX∈RD×H×W. We omit the organ indexifor simplicity
in this section.
2D image feature extraction.To extract 2D features
from each slice, we apply a ViT-Large [4] model as a 2D
feature encoderE 2D. This process is illustrated in Fig. 4(a).
Inspired by the use of the [CLS] token in ViT for global
image representation, we extract the 2D feature of slicex l

as
f2D
l=E 2D 
xcls,W·tokenize(x l)
+E2D
pos
,(5)
wherexclsis a learnable [CLS] token, and each image patch
is tokenized and linearly projected usingW. The positional
embeddingE2D
posis added to the [CLS] token and patch em-
beddings. The outputf2D
l∈Rdimgcorresponds to the final
embedding of the [CLS] token, which represents the global
feature of the slice. Each image slicex lis independently
fed into the encoder to extract its 2D feature representation.
As a result, the set of 2D features is given by
F2D=
f2D
1,f2D
2, . . . ,f2D
D
∈RD×d img.(6)
3D volume feature extraction.For 3D volume feature
extraction, we extend the ViT structure to three dimensions,
as illustrated in Fig. 4(b). While the 2D ViT processes a
single image, our model divides the entire 3D volume into
patches of size(p D, pH, pW). To preserve spatial structure,
we employ a 3D convolution [29] with a 3D kernel for patch
embedding, defined as:
X3D=Conv3D(X;kernel= (p D, pH, pW)).(7)
This operation producesD/p D×H/p H×W/p Wpatches,
each embedded into ad img-dimensional vector space. To
extract the 3D volume representation, we use a transformer-
based 3D encoder:
f3D=E 3D 
xcls,X3D
+E3D
pos
.(8)
wherexclsis a learnable [CLS] token andE3D
posis a learnable
positional embedding. The 3D encoder is designed to be
more lightweight than the 2D encoder, as volumetric data
requires higher computational resources.
Cross attention.In medical imaging, certain slices often
contain critical features of lesions or organs. Therefore,
rather than treating all slices equally, it is important to se-
lectively emphasize informative slices based on global con-
text. To this end, we employ cross-attention to integrate
slice-wise 2D features with the global 3D context, as illus-
trated in Fig. 4(c). We use the 3D global featuref 3Das the
query and the 2D featuresF 2Das the keys and values:
Fout=crossAttn(Q=f3D, K=V=F2D).(9)
This allows the model to focus more on the most informa-
tive slices. Finally, we add the features to generate inte-
grated representations by
¯Fout=F2D+Fout∈RD×d img.(10)
This fused representation ¯Foutintegrates the original 2D
slice features with globally attended featureFout, enhanc-
ing semantic understanding of both the full scan and fine-
grained local details.3.4. Loss functions
For organ classification, we apply separate MLP heads to
the 2D slice featuresF2D, 3D volume featuresf3D, and
the fused features ¯Fout. Each MLP head produces organ
class predictions and is trained using cross-entropy loss be-
tween predictions and ground truth labels. The classifica-
tion losses are defined asL 2D,L3D, andL fusion respectively,
and the total image classification loss is:
Limage=L 2D+L 3D+L fusion.(11)
Image–symptom text alignment loss.To align symp-
tom text featuresftxtand image features ¯Foutin a shared
embedding space, we optimize a contrastive loss. An addi-
tional projection layerWalign∈Rdimg×d txtis applied to the
text branch to match the image feature dimensiondimgby
ˆftxt=Walign·¯ftxt.
We define thel-th slice image feature of thei-th organ by
¯fout
il∈¯Fout
i. For effective multimodal learning, we employ
InfoNCE [22] as follows:
Lalign=−NX
i=1
logexp
avgSim( ¯fout
il,ˆftxt
ij)/τ
PN
k=1exp
avgSim( ¯fout
il,ˆftxt
kj)/τ
,
(12)
whereτis the temperature scaling factor, andi̸=k.
avgSim(·)denotes an average similarity function defined by
avgSim( ¯fout
l,ˆftxt
j) =DX
l=1NtxtX
j=1sim(¯fout
l,ˆftxt
j),(13)
wherelandjare indexes of the image slice and the symp-
tom text, respectively. sim(·)is a cosine similarity function.
The final loss is defined asL total=L image+L align, which
is optimized to train the image feature extractor and align
image and text features in a shared embedding space.
4. Experimental Results
4.1. Datasets and Settings
Datasets.For image data, we utilize the BTCV
dataset [17], which contains contrast-enhanced abdominal
CT images with 11 organ segmentation labels. Among
them, we used seven main organs: liver, pancreas, kidney,
gallbladder, spleen, stomach, and adrenal glands. We ex-
cluded non-organ structures such as the esophagus, blood
vessels, and veins. We apply the segmentation masks for
seven organs to the original CT images to generate organ-
specific image regions for our experiments. For symptom
text data, we use the organ–symptom dataset constructed as
described in Sec. 3.1 The dataset contains 1,400 symptom
descriptions (200 per organ for the selected seven organs)
and is split into training and test sets with an 8:2 ratio. The
training set uses automatically generated soft labels from

LIVER
STOMACH
SPLEEN
PANCREAS
ADRENAL GLAND
GALLBLADDER
KIDNEYPositive anchorText featureFigure 5. t-SNE visualization of symptom text embeddings
learned with soft labeling. Different colors indicate different or-
gan classes, and star symbols denote the learned positive anchors
for each organ. (Best viewed in color)
Sec. 3.2, while the test set is annotated with multi-labels
validated by medical experts.
Settings.All experiments were conducted on a NVIDIA
L40S GPU. Training parameters for the text embedding
MLP are as follows: epochs – 500, batch size – 32, learn-
ing rate – 0.001, optimizer – Adam. The margin value
for soft labeling was empirically set to 0.8. For the 2D
transformer encoder, we used an ImageNet-21k [26] pre-
trained model as backbone and fine-tuned it on the BTCV
dataset for 10 epochs. For 3D feature extraction, we set
patch size(p D, pH, pW) = (8,16,16)with six transformer
blocks. The 3D transformer and cross-attention module
were trained for 50 epochs. Training parameters for the
text-image alignment linear projection layer are as fol-
lows: epochs – 50, InfoNCE loss temperature parameter
τ– 0.1. We evaluate performance using Rank-kaccuracy
(k= 1,2,3) and mean Average Precision (mAP). Rank-k
accuracy measures the proportion of cases where the correct
answer appears in the topkpredictions, while mAP repre-
sents the average precision across all organ classes.
Inference.The cosine similarity between the learned
positive anchorsv+
iand image features ¯Foutserves as the
inference probability as shown in Fig. 7.
4.2. Effects of the Proposed Methods
Soft labeling for symptom text embeddings.To verify
that complex symptom-organ relationships are effectively
learned, we analyze the symptom text embedding space
trained with soft labeling. Figure 5 shows a t-SNE visu-
alization of text embeddings learned with soft labeling. Un-
like hard labeling which creates completely separated clus-
ters for each class, soft labeling forms flexible clusters that
reflect associations between symptoms and multiple organs.
Table 1 shows examples of symptom texts that are closest toCategory Dist Symptom text
Closest
Symptom0.74 Signs of chronic disease of the detoxifi-
cation organ including palmar erythema
and spider angiomas noted.
1.55 Presence of xanthomas on the eyelids re-
ported as a new finding.
Farthest
Symptom39.41 Severe abdominal pain in the right upper
quadrant noted.
40.31 Dull abdominal discomfort after eating
fatty foods noted by the patient.
Table 1. Closest and farthest symptom texts from the liver positive
anchorv+
1in the text embedding space.
gall-
bladder
adrenal
glandstomachkidney
spleenliver
pancreas
(a) Hard labeling (b) Soft labeling
Figure 6. Correlation heatmaps between organs learned using (a)
hard labeling and (b) soft labeling.
and farthest from the liver’s positive anchor in the embed-
ding space, measured by L2 distance. The closest symp-
toms consist of liver-specific descriptions directly related to
liver function, whereas the farthest symptoms are general
abdominal symptoms. This demonstrates that symptoms
strongly associated with a single organ tend to be located
near cluster centers, while those related to multiple organs
are more likely to appear near cluster boundaries.
Figure 6 shows correlation heatmaps between organs
learned with hard labeling and soft labeling approaches.
In hard labeling (Fig. 6(a)), each organ is learned inde-
pendently, resulting in uniformly low correlations between
organs. In contrast, soft labeling exhibits higher correla-
tions for organ pairs with anatomical proximity or func-
tional relationships, as shown in Fig. 6(b). For example,
the heatmap shows that the stomach and liver have strong
location-based associations, likely due to their anatomical
proximity and shared symptoms such as “abdominal pain”.
In contrast, the gallbladder and liver show strong functional
associations through their common role in bile production.
We evaluate the effectiveness of soft labeling in a text-
to-image matching task, where a symptom text is used to
retrieve related organs from abdominal CT images. Table 2
compares the retrieval performance of hard and soft label-
ing approaches. Soft labeling consistently outperforms hard
labeling across all metrics, with larger performance gaps in
Rank-2 and Rank-3. This suggests that hard labeling fo-
cuses on single-organ prediction. In contrast, soft labeling

Labeling Rank-1 Rank-2 Rank-3 mAP
Hard 74.64 82.14 86.42 67.53
Soft 77.50 89.64 94.85 73.46
Table 2. Comparison of hard and soft labeling in text-to-image
matching performance.
Structure Rank-1 Rank-2 Rank-3 mAP
2D Only 75.00 86.07 94.29 71.53
3D Only 33.21 60.00 74.29 39.60
2D+3D (concat) 76.43 87.14 94.64 72.56
2D+3D (cross-Attn) 77.50 89.64 94.85 73.46
Table 3. Comparison of different 2D and 3D image feature fusion
approaches. The best and second-best scores are marked in bold
and underlined.
better reflects clinical reality, where symptoms may be re-
lated to multiple organs, thereby facilitating the learning of
multi-organ relationships.
Fusion of 2D/3D image features.To evaluate the effec-
tiveness of fusing 2D slice-level details with 3D volume-
level global information, we compare various image fea-
ture fusion methods. Table 3 summarizes the performance
comparison of different 2D-3D feature fusion approaches.
Using only 2D features achieves a Rank-1 accuracy of 75%
and an mAP of 86.07%, while using only 3D features results
in significantly lower performance at 33.21% and 60%, re-
spectively. This is because 2D features are extracted in-
dependently from each slice, resulting in a larger number
of training samples, whereas 3D features are extracted at
the volume level only. Simple concatenation of 2D and 3D
features improves performance to 76.43% Rank-1 accuracy
and 72.56% mAP, compared to using either feature alone.
In contrast, our proposed 2D-3D fusion achieves the best
performance, demonstrating more effective integration of
fine-grained 2D anatomical details with 3D spatial context.
4.3. Performance Comparison
We evaluate the performance of the proposed Med-SORA
against existing methods on the symptom text-to-medical
image reasoning task, as shown in Tab. 4. As no prior
work has been explicitly designed for this task, we conduct
fair comparisons by adapting representative models from
relevant domains. These include a general text encoder:
BERT [3], medical-specific language models: Clinical-
BERT [11], PubMedBERT [8], BioMedGPT [34]. In addi-
tion, we tested multimodal vision-language models (VLM):
CLIP [25], PubMedCLIP [5], BioMedCLIP [35]. The ex-
perimental settings for adapting these baselines to our task
are described in detail in the supplementary materials to en-
sure fairness.
Text-based models–BERT, ClinicalBERT, PubMed-
BERT, and BioMedGPT–were combined with Med-SORA’s
image encoder and fine-tuned for the symptom-to-organ
matching task. We evaluated both zero-shot and fine-tunedModel Rank-1 Rank-2 Rank-3 mAP
BERT [3] 60.71 81.43 87.86 51.19
ClinicalBERT [11] 64.29 85.00 91.43 55.58
PubMedBERT [8] 74.29 85.00 90.71 71.65
BioMedGPT [34] 38.21 62.50 78.92 37.40
CLIP [25] 60.71 73.57 85.71 53.49
PubMedCLIP* [5] 47.85 60.35 83.57 38.12
PubMedCLIP [5] 63.93 79.29 88.21 61.43
BioMedCLIP* [35] 18.93 37.50 55.36 20.02
BioMedCLIP [35] 23.21 40.00 54.29 23.95
Ours 77.50 89.64 94.85 73.46
Table 4. Performance comparison with existing text and multi-
modal models on symptom-to-organ reasoning task. * denotes
zero-shot performance.
performance for VLM-based models, especially PubMed-
CLIP and BioMedCLIP, which have demonstrated strong
zero-shot performance in prior studies. Our method, Med-
SORA achieves superior performance across all evaluation
metrics compared to other methods.
Figure 7 illustrates the reasoning results of Med-SORA
when given symptom text queries. The model infers the
most relevant organ and highlights its 3D segmentation
based on the symptom description. As shown in Fig. 7(a),
when a symptom query is related to multiple organs, the
model assigns probabilities to multiple relevant organs. In
contrast, symptoms that are strongly associated with a spe-
cific organ, as in Fig. 7(b), result in a distinctly high prob-
ability for that organ. When irrelevant symptom queries
such as “knee joint pain” or “headache” are given as in-
put, as shown in Fig. 7(c), all abdominal organs receive uni-
formly low probability scores. These results demonstrate
that Med-SORA effectively handles both single-organ and
multi-organ symptom reasoning, while remaining robust to
out-of-domain queries without producing false positives.
The intuitive 3D visualization helps clinicians quickly
understand symptom-organ associations and can serve as
an effective educational tool in clinical settings. In addi-
tion, it may assist in patient communication by providing
visual explanations of symptom-related organ involvement.
Further potential applications and directions for future work
are discussed in Sec. 5.
5. Conclusions and Future Work
In this paper, we proposed Med-SORA, a framework for
symptom-to-organ reasoning that infers symptom-related
organs in abdominal CT images from patient symptom text.
We constructed a RAG-based dataset from medical litera-
ture and applied soft labeling with learnable organ anchors
to model one-to-many symptom–organ relationships. To
understand anatomical structures more effectively, we de-
signed a 2D–3D cross-attention architecture that integrates
both local and global CT features. Experiments show that
Med-SORA outperforms existing multimodal medical mod-

Restless legs syndrome reported, particularly during dialysis or in later stages of renal failure
(b) Abdominal organ specific query examplesPain in the knee jointSevere headache and migraine symptoms
(c) Non-abdominal organ query examples(a) Abdominal organ relevant query examplesChronic dull ache in the rightupper quadrant reported,worsening over timeSigns of gastrointestinalbleeding, such as melena,reportedAltered appetite characterizedby a preference for blandfoods observed1.0Probability
0.0
yxzyxzyxzkidneystomachstomachliverlivergallbladderpancreas:Figure 7. 3D probability visualizations generated by Med-SORA for different symptom text queries (shown in gray boxes): (a) symptoms
related to multiple organs, (b) organ-specific symptoms, and (c) non-abdominal symptoms. Ground-truth organs are indicated with overlaid
star⋆markers. Med-SORA effectively handles both single-organ and multi-organ symptom reasoning.
Symptom Text: Acute onset of severe right upper quadrant pain after eating fatty food documented
(a) Original CT Sagittal view 
(b) Organ Reasoning (c) Anomaly Detection [Score]
 liver: 0.949
 pancreas: 0.920 
 stomach: 0.838
Figure 8. Future work directions for Med-SORA.
els and enables interpretable 3D visualizations aligned with
clinical reasoning.
This study has several limitations that suggest directions
for future work. First, Med-SORA learns each organ inde-
pendently, which limits holistic understanding of anatomi-
cal context. Second, the current framework provides only
organ-level reasoning without capturing fine-grained organ
details. Future extensions include joint multi-organ learn-
ing, detailed visualization, and automatic anomaly detec-
tion within related organ regions. These improvements
could enhance clinical utility, and examples of such exten-
sions are illustrated in Fig. 8.
References
[1] https://pubmed.ncbi.nlm.nih.gov/. 1996. 3
[2] https://www.wikipedia.org/. 2001. 3
[3] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. Bert: Pre-training of deep bidirectional trans-
formers for language understanding. InProceedings of the
2019 conference of the North American chapter of the asso-
ciation for computational linguistics: human language tech-
nologies, volume 1 (long and short papers), pages 4171–
4186, 2019. 7
[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, G Heigold, S Gelly,
et al. An image is worth 16x16 words: Transformers for
image recognition at scale. InInternational Conference on
Learning Representations, 2020. 4
[5] Sedigheh Eslami, Christoph Meinel, and Gerard De Melo.
Pubmedclip: How much does clip benefit visual questionanswering in the medical domain? InFindings of the As-
sociation for Computational Linguistics: EACL 2023, pages
1181–1193, 2023. 1, 2, 7
[6] Yunhe Gao. Training like a medical resident: Context-prior
learning toward universal medical image segmentation. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 11194–11204, 2024. 1
[7] Yuting Gao, Jinfeng Liu, Zihan Xu, Tong Wu, Enwei Zhang,
Ke Li, Jie Yang, Wei Liu, and Xing Sun. Softclip: Softer
cross-modal alignment makes clip stronger. InProceed-
ings of the AAAI Conference on Artificial Intelligence, pages
1860–1868, 2024. 3
[8] Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto
Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng Gao,
and Hoifung Poon. Domain-specific language model pre-
training for biomedical natural language processing.ACM
Transactions on Computing for Healthcare (HEALTH), 3(1):
1–23, 2021. 7
[9] Elad Hoffer and Nir Ailon. Deep metric learning using triplet
network. InSimilarity-based pattern recognition: third inter-
national workshop, SIMBAD 2015, Copenhagen, Denmark,
October 12-14, 2015. Proceedings 3, pages 84–92. Springer,
2015. 4
[10] Yutao Hu, Tianbin Li, Quanfeng Lu, Wenqi Shao, Junjun He,
Yu Qiao, and Ping Luo. Omnimedvqa: A new large-scale
comprehensive evaluation benchmark for medical lvlm. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 22170–22183, 2024. 1
[11] Kexin Huang, Jaan Altosaar, and Rajesh Ranganath. Clini-
calbert: Modeling clinical notes and predicting hospital read-
mission.arXiv preprint arXiv:1904.05342, 2019. 7
[12] Ta Duc Huy, Sen Kim Tran, Phan Nguyen, Nguyen Hoang
Tran, Tran Bao Sam, Anton van den Hengel, Zhibin Liao,
Johan W Verjans, Minh-Son To, and Vu Minh Hieu Phan.
Interactive medical image analysis with concept-based simi-
larity reasoning. InProceedings of the Computer Vision and
Pattern Recognition Conference, pages 30797–30806, 2025.
2
[13] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen,
and Xinghua Lu. Pubmedqa: A dataset for biomedical re-
search question answering. InProceedings of the 2019 Con-
ference on Empirical Methods in Natural Language Process-
ing and the 9th International Joint Conference on Natural

Language Processing (EMNLP-IJCNLP), pages 2567–2577,
2019. 2
[14] Qiao Jin, Won Kim, Qingyu Chen, Donald C Comeau, Lana
Yeganova, W John Wilbur, and Zhiyong Lu. Medcpt: Con-
trastive pre-trained transformers with large-scale pubmed
search logs for zero-shot biomedical information retrieval.
Bioinformatics, 39(11):btad651, 2023. 2
[15] Hanbin Ko and Chang-Min Park. Bringing clip to the clinic:
Dynamic soft labels and negation-aware learning for medical
analysis. InProceedings of the Computer Vision and Pattern
Recognition Conference, pages 25897–25906, 2025. 3
[16] Anastasia Krithara, Anastasios Nentidis, Konstantinos
Bougiatiotis, and Georgios Paliouras. Bioasq-qa: A manu-
ally curated corpus for biomedical question answering.Sci-
entific Data, 10(1):170, 2023. 2
[17] Bennett Landman, Zhoubing Xu, Juan Igelsias, Martin
Styner, Thomas Langerak, and Arno Klein. Miccai multi-
atlas labeling beyond the cranial vault–workshop and chal-
lenge. InProc. MICCAI multi-atlas labeling beyond cra-
nial vault—workshop challenge, page 12. Munich, Germany,
2015. 2, 5
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems,
33:9459–9474, 2020. 2, 3
[19] Fenglin Liu, Chenyu You, Xian Wu, Shen Ge, Xu Sun, et al.
Auto-encoding knowledge graph for unsupervised medical
report generation.Advances in Neural Information Process-
ing Systems, 34:16266–16279, 2021. 1
[20] Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford,
Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas
Tezak, Jong Wook Kim, Chris Hallacy, et al. Text and
code embeddings by contrastive pre-training.arXiv preprint
arXiv:2201.10005, 2022. 3
[21] Harsha Nori, Mayank Daswani, Christopher Kelly, Scott
Lundberg, Marco Tulio Ribeiro, Marc Wilson, Xiaoxuan
Liu, Viknesh Sounderajah, Jonathan Carlson, Matthew P
Lungren, et al. Sequential diagnosis with language models.
arXiv preprint arXiv:2506.22405, 2025. 2
[22] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Repre-
sentation learning with contrastive predictive coding.arXiv
preprint arXiv:1807.03748, 2018. 5
[23] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan
Sankarasubbu. Medmcqa: A large-scale multi-subject multi-
choice dataset for medical domain question answering. In
Conference on health, inference, and learning, pages 248–
260. PMLR, 2022. 2
[24] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya
Sutskever, et al. Improving language understanding by gen-
erative pre-training. 2018. 2
[25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. InInternational conference on machine learning, pages
8748–8763. PmLR, 2021. 2, 7[26] Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, and Lihi
Zelnik-Manor. Imagenet-21k pretraining for the masses.
arXiv preprint arXiv:2104.10972, 2021. 6
[27] Kody Shaw, Marcus A Henning, and Craig S Webster. Ar-
tificial intelligence in medical education: a scoping review
of the evidence for efficacy and future directions.Medical
Science Educator, pages 1–14, 2025. 2
[28] Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery
Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R
Pfohl, Heather Cole-Lewis, et al. Toward expert-level med-
ical question answering with large language models.Nature
Medicine, pages 1–8, 2025. 2
[29] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani,
and Manohar Paluri. Learning spatiotemporal features with
3d convolutional networks. InProceedings of the IEEE inter-
national conference on computer vision, pages 4489–4497,
2015. 5
[30] Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu,
Filippo Menolascina, Yueming Jin, and Vicente Grau. Medi-
cal graph rag: Evidence-based medical large language model
via graph retrieval-augmented generation. InProceedings of
the 63rd Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), pages 28443–
28467, 2025. 2
[31] Zibo Xu, Qiang Li, Weizhi Nie, Weijie Wang, and Anan
Liu. Structure causal models and llms integration in medical
visual question answering.IEEE Transactions on Medical
Imaging, 2025. 2
[32] Xiaowei Yu, Zihao Wu, Lu Zhang, Jing Zhang, Yanjun Lyu,
and Dajiang Zhu. Cp-clip: Core-periphery feature alignment
clip for zero-shot medical image analysis. InInternational
Conference on Medical Image Computing and Computer-
Assisted Intervention, pages 88–97. Springer, 2024. 1
[33] Cyril Zakka, Rohan Shad, Akash Chaurasia, Alex R
Dalal, Jennifer L Kim, Michael Moor, Robyn Fong,
Curran Phillips, Kevin Alexander, Euan Ashley, et al.
Almanac—retrieval-augmented language models for clinical
medicine.Nejm ai, 1(2):AIoa2300068, 2024. 2
[34] Kai Zhang, Rong Zhou, Eashan Adhikarla, Zhiling Yan,
Yixin Liu, Jun Yu, Zhengliang Liu, Xun Chen, Brian D Davi-
son, Hui Ren, et al. A generalist vision–language foundation
model for diverse biomedical tasks.Nature Medicine, pages
1–13, 2024. 7
[35] Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu,
Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao,
Mu Wei, Naveen Valluri, et al. Biomedclip: a multimodal
biomedical foundation model pretrained from fifteen million
scientific image-text pairs.arXiv preprint arXiv:2303.00915,
2023. 1, 2, 7