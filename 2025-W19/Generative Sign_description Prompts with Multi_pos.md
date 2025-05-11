# Generative Sign-description Prompts with Multi-positive Contrastive Learning for Sign Language Recognition

**Authors**: Siyu Liang, Yunan Li, Wentian Xin, Huizhou Chen, Xujie Liu, Kang Liu, Qiguang Miao

**Published**: 2025-05-05 00:57:57

**PDF URL**: [http://arxiv.org/pdf/2505.02304v1](http://arxiv.org/pdf/2505.02304v1)

## Abstract
Sign language recognition (SLR) faces fundamental challenges in creating
accurate annotations due to the inherent complexity of simultaneous manual and
non-manual signals. To the best of our knowledge, this is the first work to
integrate generative large language models (LLMs) into SLR tasks. We propose a
novel Generative Sign-description Prompts Multi-positive Contrastive learning
(GSP-MC) method that leverages retrieval-augmented generation (RAG) with
domain-specific LLMs, incorporating multi-step prompt engineering and
expert-validated sign language corpora to produce precise multipart
descriptions. The GSP-MC method also employs a dual-encoder architecture to
bidirectionally align hierarchical skeleton features with multiple text
descriptions (global, synonym, and part level) through probabilistic matching.
Our approach combines global and part-level losses, optimizing KL divergence to
ensure robust alignment across all relevant text-skeleton pairs while capturing
both sign-level semantics and detailed part dynamics. Experiments demonstrate
state-of-the-art performance against existing methods on the Chinese SLR500
(reaching 97.1%) and Turkish AUTSL datasets (97.07% accuracy). The method's
cross-lingual effectiveness highlight its potential for developing inclusive
communication technologies.

## Full Text


<!-- PDF content starts -->

Generative Sign-description Prompts with Multi-positive
Contrastive Learning for Sign Language Recognition
Siyu Liang
syliang_233@stu.xidian.edu.cn
Xidian University
Xi’an, Shaanxi, ChinaYunan Li
yunanli@xidian.edu.cn
Xidian University
Xi’an, Shaanxi, ChinaWentian Xin
wtxin@dlmu.edu.cn
Dalian Martime University
Dalian, Liaoning, China
Huizhou Chen
chenhz@stu.xidian.edu.cn
Xidian University
Xi’an, Shaanxi, ChinaXujie Liu
24031110049@stu.xidian.edu.cn
Xidian University
Xi’an, Shaanxi, ChinaKang Liu
kangliu@stu.xidian.edu.cn
Xidian University
Xi’an, Shaanxi, China
Qiguang Miao∗
qgmiao@xidian.edu.cn
Xidian University
Xi’an, Shaanxi, China
(3) Multi -positive contrastive learning (2) Single -positive contrastive learning (1) Description Generation based on LLM
< Label >
Global descriptionPart-specific descriptions
: Global description. Part-specific descriptions include      : Left hand;       : Right hand ;      : Face;      : Mouth;      : Body
→←: Pulling similar samples closer.            ←→: Pushing dissimilar samples apart.
Global Feature
PositiveNegatives• Face
• Left hand
• Mouth
• Body
Partial Feature• Right hand
PositivesNegatives
Figure 1: Multi-positive contrastive learning for LLMs generated multipart sign-description.
Abstract
Sign language recognition (SLR) faces fundamental challenges in
creating accurate annotations due to the inherent complexity of
simultaneous manual and non-manual signals. To the best of our
knowledge, this is the first work to integrate generative large lan-
guage models (LLMs) into SLR tasks. We propose a novel Generative
Sign-description Prompts Multi-positive Contrastive learning ( GSP-
MC) method that leverages retrieval-augmented generation (RAG)
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
MM ’25, Dublin, Ireland
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/XXXXXXX.XXXXXXXwith domain-specific LLMs, incorporating multi-step prompt en-
gineering and expert-validated sign language corpora to produce
precise multipart descriptions. The GSP-MC method also employs
a dual-encoder architecture to bidirectionally align hierarchical
skeleton features with multiple text descriptions (global, synonym,
and part level) through probabilistic matching. Our approach com-
bines global and part-level losses, optimizing KL divergence to en-
sure robust alignment across all relevant text-skeleton pairs while
capturing both sign-level semantics and detailed part dynamics.
Experiments demonstrate state-of-the-art performance against ex-
isting methods on the Chinese SLR500 (reaching 97.1%) and Turk-
ish AUTSL datasets (97.07% accuracy). The method’s cross-lingual
effectiveness highlight its potential for developing inclusive com-
munication technologies.arXiv:2505.02304v1  [cs.CL]  5 May 2025

MM ’25, October 27–30, 2025, Dublin, Ireland Siyu Liang, et al.
CCS Concepts
•Computing methodologies →Learning latent representa-
tions ;Computer vision representations ;Computer vision prob-
lems .
Keywords
Sign Language Recognition, Contrastive Learning, Generative Large
Language Model, Modality Fusion
1 Introduction
Due to its importance in bridging communication gaps across di-
verse human communities, sign language recognition (SLR) has
attracted substantial research interest. The advent of high-precision
depth sensors, including Kinect [ 25] and RealSense [ 13], coupled
with advances in pose estimation algorithms [ 8,30], , has signifi-
cantly simplified the acquisition of human joint positions in recent
years. Skeleton-based SLR methods, which rely on body joint move-
ments, have gained considerable attention due to their computa-
tional efficiency and robustness to variations in lighting conditions,
viewpoints, and background noise. An increasing number of SLR
works are based on the skeleton modality [ 21] or use skeletons as
one of the multimodal inputs [16, 20, 38, 39].
The remarkably rapid development of pre-trained generative
LLMs has led to their expanding applications across various recog-
nition domains [ 1,23,36], particularly in action recognition. Ap-
proaches such as GAP [ 33] and SpaMo [ 12] have demonstrated
how LLMs can enhance recognition by generating fine-grained
textual descriptions or interpreting spatial-motion features. Al-
though these methods primarily employ LLMs as sophisticated text
processors, their generative capabilities remain underexplored for
domain-specific applications like sign language recognition. This
presents unique challenges, as sign language processing requires
both linguistic precision and specialized domain knowledge that
general-purpose LLMs typically lack. Potential solutions must ad-
dress the fundamental tension between domain-specific motion
accuracy and linguistic expressiveness in generated descriptions.
Although prompt engineering has enabled LLMs to assist in
action recognition by generating auxiliary text descriptions, such
approaches face unique hurdles in SLR. Sign language requires
expert knowledge, as subtle variations in hand shape, motion, or
expression convey different meanings. General LLMs often produce
descriptions that are either overly generic or semantically incon-
sistent. Existing methods [ 33,35], designed for action recognition,
struggle with inaccuracies and hallucinations in sign descriptions.
This limitation calls for domain-aware prompting techniques that
harmonize LLMs’ generative flexibility with the structural preci-
sion demanded by sign language. This suggests the need for new
prompting paradigms that can bridge the generative capacity of
LLMs with the strict descriptions of expert-defined signs.
Contrastive learning has revolutionized unsupervised repre-
sentation learning in multiple domains. MoC [ 9] implements a
momentum-based dynamic dictionary with a queue and moving-
average encoder for scalable contrastive learning. SimCL [ 2] pro-
poses a method for text-to-image models, treating multiple images
generated from the same text prompt as positive samples. MC [ 22]
proposes a multi-view enhanced contrastive learning method forvisual representation by maximizing agreements between multi-
view radiographs and their corresponding reports, solving medical
imaging issues. Sign language recognition challenges contrastive
learning with many-to-many relationships: One sign has multi-
ple valid descriptions (all of which should be treated as positives),
and descriptions often focus on partial actions. The single positive
contrastive learning method struggle with these variable, incom-
plete positives, requiring new approaches that handle probabilistic
alignments while preserving discrimination.
The contributions of our proposed method are as follows.
1. To the best of our knowledge, we are the first to integrate
generative LLMs into SLR through our Generative Sign-description
Prompts (GSP). GSP employs retrieval-augmented generation with
domain-specific LLMs to produce accurate multipart sign descrip-
tions, providing reliable textual grounding for contrastive learning.
2. We design the Multi-positive Contrastive learning (MC) ap-
proach, which combines retrieval-augmented generative descrip-
tions from expert-validated knowledge bases and a novel multi-
positive contrastive learning paradigm.
3. Comprehensive experiments on the Chinese SLR500 and Turk-
ish AUTSL datasets further validate the effectiveness of our method,
achieving state-of-the-art accuracy (97.1% and 97.07% respectively).
The consistent and robust performance across languages demon-
strates generalization capabilities.
2 Related Works
2.1 Skeleton-Based Sign Language Recognition
Current research in skeleton-based SLR has explored several promis-
ing approaches. Self-supervised learning approaches, such as Sign-
BER [ 10] and BES [ 39], employ masked sequence modeling and
quantized pseudo-labels to learn sign representations from hand
pose sequences. These methods demonstrate the potential for learn-
ing sign representations without extensive annotations, although
they primarily focus on low-level motion patterns rather than lin-
guistic semantics. Graph-based architectures like CoSig [ 16] utilize
specialized graph convolutional networks to capture the skeletal
dynamics, but often neglect the global semantic relationships in-
herent in the sign language. Meanwhile, multimodal pretraining
methods such as MAS [ 38] integrate motion-aware autoencoders
with semantic alignment to learn joint visual-linguistic representa-
tions. Despite these advances, domain methods exhibit persistent
limitations by treating sign language primarily as motion sequences
while neglecting its structured linguistic nature. This leads to exces-
sive dependence on data-driven motion patterns, weak integration
with linguistic knowledge, and inadequate visual-semantic align-
ment, which are fundamental shortcomings that hinder linguisti-
cally grounded SLR systems. Thus, our work addresses these gaps by
multi-positive contrastive learning, aligning each sign with multiple
expert-verified descriptions while maintaining cross-modal discrim-
ination via fixed-text-encoder training. This approach learns robust
visual features that absorb natural sign variations without compu-
tational overhead during inference, achieving more linguistically-
grounded recognition than previous methods.

Generative Sign-description Prompts with Multi-positive Contrastive Learning for Sign Language Recognition MM ’25, October 27–30, 2025, Dublin, Ireland
Skeleton 
Encoder𝑬𝑬𝒔𝒔
Key points𝑆𝑆
Skeleton 
Features
𝓟𝓟𝟏𝟏: Describe 
the sign < > …Part-specific Skeleton Encoder Description -Driven Sign Language 
Representation Learningcls
< label >𝑇𝑇𝑏𝑏
: Knowledge base. :Trainable parameters. :Frozen parameters.𝓟𝓟𝟐𝟐:The sign < >’s synonym.𝑇𝑇𝑠𝑠
𝓟𝓟𝟑𝟑:Refine the sign …𝑇𝑇𝑔𝑔
𝓟𝓟𝟒𝟒:Decompose the sign 
into multipart and … 𝑇𝑇𝑝𝑝𝑻𝑻Global Average Pooling 𝑺𝑺𝑔𝑔
Text Encoder 𝑬𝑬𝒕𝒕Face part average pooling 𝑺𝑺𝑓𝑓
Left hand part average pooling 𝑺𝑺𝑙𝑙𝑙
Right hand part average pooling 𝑺𝑺𝑟𝑟𝑙
Body part average pooling 𝑺𝑺𝑏𝑏
Mouth part average pooling 𝑺𝑺𝑚𝑚
Generative Sign -description Promptsmulti
con
𝒔𝒔1
𝒔𝒔2
𝒔𝒔𝐵𝐵 …𝒔𝒔1
𝒔𝒔2…
𝒕𝒕𝟏𝟏𝟏𝟏𝒕𝒕𝟏𝟏𝟐𝟐𝒕𝒕𝟐𝟐𝟏𝟏𝒕𝒕𝑩𝑩𝟏𝟏𝒕𝒕𝑩𝑩𝟐𝟐 𝒕𝒕𝟏𝟏𝟐𝟐𝒕𝒕𝑩𝑩𝟑𝟑…𝒕𝒕𝟐𝟐𝟏𝟏𝒕𝒕𝑩𝑩𝟏𝟏𝒕𝒕𝑩𝑩𝟐𝟐𝒔𝒔1𝒕𝒕𝟏𝟏𝟏𝟏
𝒔𝒔𝐵𝐵𝒕𝒕𝑩𝑩𝟐𝟐𝒔𝒔𝐵𝐵𝒕𝒕𝑩𝑩𝟑𝟑𝒔𝒔2𝒕𝒕𝟐𝟐𝟏𝟏𝒔𝒔1𝒕𝒕𝟏𝟏𝟐𝟐
𝒔𝒔𝐵𝐵𝒕𝒕𝑩𝑩𝟏𝟏
: Large language model.
𝒕𝒕𝟏𝟏𝟏𝟏𝒔𝒔𝐵𝐵
Figure 2: The overall architecture of GSP-MC method. S𝑔denotes global skeleton features for classification, S𝑓, etc. represents
part-specific features for contrastive learning, and Tis encoded text features. Training is guided by L𝑐𝑙𝑠andL𝑐𝑜𝑛losses.
2.2 Text-Augmented Sign Language
Representation
Recent progress in combining textual and visual features for SLR
predominantly adopt two distinct strategies. On the one hand, man-
ual text encoding methods such as NLA-SLR [ 41] and𝐶2𝑆𝑇[37]
leverage predefined gloss vocabularies, but are constrained by static
description sets that fail to capture execution variations and face
scalability issues due to high manual annotation costs. On the other
hand, generative text integration approaches, including Action-
GPT [ 17] and GAP [ 33], employ LLM-based action description gen-
eration. However, these primarily target generic action recognition
or vision-language tasks, leaving key SLR challenges unaddressed:
hallucinated sign descriptions, lack of sign-specific linguistic con-
straints, and difficulties in modeling simultaneous sign compo-
nents (e.g., hand shape, motion, and orientation). Notably, while
Sign2GPT [ 32] attempts integrate LLM, current methods still lack
robust generative LLM solutions for these domain-specific require-
ments. Our proposed approach tackles these issues through RAG to
anchor LLM outputs in knowledge bases and multi-level visual-text
alignment constrained by various generated descriptions.
3 Methods
The Generative Sign-description Prompts with Multi-positive Con-
trastive Learning (GSP-MC) method augments skeleton-based sign
language recognition through multimodal representation learning.
As illustrated in Figure 2, the architecture comprises three elements:
(1) a group-specific skeleton encoder extracting hierarchical motion
features, (2) a generative sign-description prompts (GSP) module
producing expert knowledge grounded text descriptions, and (3) a
multi-positive contrastive learning mechanism( MC) aligning visual
and textual representations. The method maintains computational
efficiency during inference using only the skeleton encoder.The skeleton encoder 𝐸𝑠extracts both the global skeleton fea-
tures S𝑔and the part-specific features S𝑝from the input pose se-
quences. The GSP employs LLMs to produce descriptive texts 𝑡,
which are subsequently encoded into text features Tby the text
encoder𝐸𝑡. These multimodal features are then optimized through
our proposed multi-positive contrastive learning approach, which
enhances the model’s capacity to capture fine-grained action se-
mantics from textual descriptions.
3.1 Part-specific Skeleton Encoder
Our model processes input skeleton sequences 𝑆∈R𝐵×3×𝑁×𝑇as
input, where 𝐵is the batch size, 3represents (x,y,confidence), 𝑁
denotes the number of joints, and 𝑇indicates the temporal sequence
length. The model predicts the labels 𝑙∈R𝐵1as output.
Keypoint Selection and Grouping. Using HR-Net [ 30] as our
skeleton estimator, we extract 87 keypoints per frame, strategically
divided into five anatomical groups: 15 for the body 𝑆𝑏, 21 for
each hand𝑆𝑙ℎand𝑆𝑟ℎ), 10 for the mouth 𝑆𝑚and 20 for the face
𝑆𝑓, respectively. We provide a detailed analysis of this keypoint
selection strategy in Section 4.3.3.
Skeleton Encoder. Representing the skeleton as a graph 𝑆=
{𝑉,𝐸}with joints𝑉and edges𝐸, we process each part 𝑃through
layered graph convolutions.
S𝑃,𝑙+1=𝜎
D−1
2AD−1
2S𝑃,𝑙Θ𝑙
(1)
where, D∈R𝑁×𝑁is the degree matrix, Athe adjacency matrix ,
Θ𝑙∈R𝐶𝑙×𝐶𝑙+1the learnable parameter at layer 𝑙, and𝜎the activa-
tion function.
Our basic block employs multiple CTR-GC block [ 3], with each
block integrating a Channel-wise Topology Refinement Graph Con-
volution layer. We fuse channel-wise topology refinement with
graph convolution to yield the final part-specific representation.

MM ’25, October 27–30, 2025, Dublin, Ireland Siyu Liang, et al.
Table 1: Alignment results comparing primary 𝑇𝑏and refined 𝑇𝑔descriptions, highlighting: sign label, substitute descriptions,
expert-validated knowledge, primary descriptions 𝑇𝑏, and refined descriptions 𝑇𝑔.
Primary descriptions 𝑇𝑏 Expert-validated knowledge Refined descriptions 𝑇𝑔
Devoted: (I) Make the sign for “love”. (II)
Extend the thumb with one hand and
place it on the palm of the other hand,
then raise it upwards.The sign for “love”: Gently caress the
back of the thumb with one hand, ex-
pressing a feeling of “tenderness”.Devoted: (1) Gently caress the back of the thumb
with one hand, expressing a feeling of “tenderness”.
(2) Extend your thumb with one hand, sit on the other
palm, and lift it up.
Ambience: (1) One hand forms the man-
ual sign “Q”, with the fingertips point-
ing inward, placed at the nostrils. (2)
Extend your index finger with one hand
and make a big circle with your finger-
tips facing down.The manual sign “Q”: One hand with the
right thumb down, the index and mid-
dle fingers together on top, the thumb,
index, and middle fingers pinched to-
gether, the fingertips pointing forward
and slightly to the left, the ring and little
fingers bent, the fingertips touching the
palm.Ambience: (1) One hand with the right thumb down,
the index and middle fingers together on top, the
thumb, index, and middle fingers pinched together,
the fingertips pointing forward and slightly to the left,
the ring and little fingers bent, the fingertips touching
the palm. The fingertips are pointing inward, placed
at the nostrils. (2) Extend index finger with one hand
and make a circle with your fingertips facing down.
Skeleton Classification. The model optimizes a standard cross-
entropy loss:
L𝑐𝑙𝑠=−Ylog𝑝𝜃(S𝑔) (2)
where Ydenotes ground truth labels, S𝑔the global skeleton features,
and𝑝𝜃(𝑥)the predicted probability distribution.
3.2 Generative Sign-description Prompts (GSP)
The Generative Sign-description Prompts (GSP) establishes a sys-
tematic approach for generating accurate and comprehensive sign
language descriptions through advanced language model tech-
niques. As illustrated in Figure 3, the system integrates domain-
specific knowledge from authoritative sources including official
sign definitions and expert textbooks, with a capacity supporting
up to 1GB of domain-specific data and individual documents up to
50MB ( 15 million Chinese characters). This extensive knowledge
base serves as the foundation for addressing key challenges in sign
language description generation.
LLM User Promptdatabases
Enhanced 
responseRetrievalDomain -specific knowledge
Figure 3: Architecture of the customized LLM for sign de-
scription generation.
3.2.1 Expert Dataset Construction and Standardization. The method
addresses the inherent challenges of professional sign language de-
scriptions through a dual-path generation mechanism. Professional
sign language definitions frequently employ substituted descrip-
tions that create barriers for machine interpretation, manifestingas standardized manual alphabet references (“uses the ‘5’ hand-
shape”) or cross-sign equivalencies (“identical to the sign ‘good’”).
To overcome this, the system implements a rigorous standardiza-
tion process where primary descriptions 𝑇𝑏are generated through
RAG-augmented generation anchored in authoritative sources:
𝑇𝑏=LLM RAG(label,P1) (3)
Simultaneously, the method incorporates controlled diversity
through synonym variations 𝑇𝑠generated by leveraging the LLM’s
creative potential within carefully designed constraints:
𝑇𝑠=LLM(label,P2) (4)
The prompt templates P1,2incorporate domain-specific instruc-
tion tuning to ensure outputs maintain both factual accuracy through
expert grounding and linguistic diversity. For instance, a base de-
scription “palm pushes forward” might be expanded to “arm extends
with palm facing outward” while preserving core features. This dual
method effectively balances the need for standardization against the
requirement for varied expressions to enhance model robustness.
3.2.2 Redundancy Elimination and Multi-part Decomposition. The
method implements advanced processing to address extraneous in-
formation common in professional sign language materials. Through
targeted prompt design P4, the system automatically filters ir-
relevant content such as homophonic explanations (“‘One’ is ho-
mophonous with ‘idea’”), focusing exclusively on action-relevant
descriptions. As show in Table 1, the refinement process transforms
initial outputs 𝑇𝑏into complete formulations 𝑇𝑔while preserving
expert-validated knowledge:
𝑇𝑔=LLM RAG(𝑇𝑏,P3) (5)
Furthermore, the system introduces innovative multi-part de-
composition, automatically generating part-specific texts 𝑇𝑝with
corresponding anatomical annotations:
𝑇𝑝=LLM(𝑇𝑔,P4) (6)

Generative Sign-description Prompts with Multi-positive Contrastive Learning for Sign Language Recognition MM ’25, October 27–30, 2025, Dublin, Ireland
𝑻𝑻𝒑𝒑𝒑𝒑
𝑻𝑻𝒑𝒑𝒑𝒑
𝑻𝑻𝒑𝒑𝒑𝒑
𝑻𝑻𝒑𝒑𝒑𝒑 …𝑷𝑷𝒑𝒑
𝑷𝑷𝒑𝒑
𝑷𝑷𝒑𝒑
𝑷𝑷𝟓𝟓𝑷𝑷𝟒𝟒Body
Right HandLeft Hand
Face
Mouth
𝑻𝑻𝒈𝒈
𝓟𝓟𝒓𝒓𝒓𝒓𝒑𝒑𝒑𝒑𝒓𝒓
Figure 4: Generation of multipart texts.
The structured output 𝑇𝑝pairs each text segment ( 𝑇𝑝1,𝑇𝑝2, etc.)
with its corresponding sign language components. As demonstrated
in Figure 4,𝑇𝑝1describes the movement involving both 𝑝𝑎𝑟𝑡 1and
𝑝𝑎𝑟𝑡 2, while𝑇𝑝2relates to𝑝𝑎𝑟𝑡 2and𝑝𝑎𝑟𝑡 3, which establish a many-
to-many mapping between text and human body parts.
3.2.3 Text Encoder. The text encoding component employs the
CLIP text encoder as its backbone 𝐸𝑡, processing three distinct
text modalities: global descriptions 𝑇𝑔, semantically-rich synonym
variants𝑇𝑠, and fine-grained part-specific texts 𝑇𝑝. The encoding
process begins with standard tokenization, followed by process-
ing through 12 transformer blocks with multi-head self-attention
mechanisms. This architecture generates context-aware hierarchi-
cal linguistic representations that are efficiently aggregated into
fixed-dimensional feature vectors Tg,Ts, and Tp.
For part-specific encoding, individual body part descriptions un-
dergo independent processing through identical architectural com-
ponents, ensuring feature space consistency across all modalities.
The use of frozen pre-trained weights maintains established lin-
guistic knowledge while providing computational efficiency during
training. This design choice allows the method to focus innovation
on the description generation aspects while leveraging proven text
encoding capabilities.
3.3 Description-Driven Sign Language
Representation Learning
3.3.1 Text-Conditioned Multi-Positive Alignment. Our method uses
a dual-encoder architecture. It includes a skeleton encoder 𝐸𝑠and a
text encoder 𝐸𝑡.𝐸𝑠processes skeletal motion data S∈R(𝐵×3×𝑁×𝑇),
while𝐸𝑡handles action descriptions. Unlike traditional one-hot
label supervision, the proposed approach uses natural language
descriptions, which provide richer supervision for skeleton classifi-
cation. As illustrated in Figure 1, our method simultaneously aligns
each skeleton sample with multiple relevant text descriptions while
maintaining separation from negative samples.
In the global description scenario, where each skeleton sequence
corresponds to one global description 𝑇𝑔or one synonym variant 𝑇𝑠,
the number of text features 𝑀exactly matches the batch size 𝐵(i.e.,
𝑀=𝐵). This configuration reduces to conventional single-positive
contrastive learning, with each skeleton feature contrasted against
its single paired text feature.
In the part-specific description scenario, the method demon-
strates its full capability when processing part-specific descriptions
𝑇𝑝. Here, multiple textual descriptions (ranging from 0 to 𝑚perbody part) are generated for each skeleton sequence, resulting in
the total text features of 𝑀=Í𝐵
𝑖=1𝑚𝑖, where𝑀>𝐵. This ex-
panded correspondence enables genuine multi-positive contrastive
learning, as each skeleton feature can simultaneously align with
multiple relevant text descriptions capturing different aspects of
the same sign action. While 𝑇𝑔and𝑇𝑠maintain the 𝑀=𝐵cor-
respondence, the part-specific 𝑇𝑝descriptions create the 𝑀>𝐵
scenario that drives our multi-positive learning advantage. This
hierarchical text representation allows the model to simultaneously
learn both holistic sign semantics and fine-grained part dynamics.
The dual-encoders are jointly optimized by contrasting skeleton-
text pairs in two directions within the batch:
q𝑠→𝑡
𝑖𝑎(s)=exp(sim(s𝑖,t𝑖𝑗)/𝜏)
Í𝐵
𝑖=1Í𝑚
𝑘=1exp(sim(s𝑖,t𝑖𝑘)/𝜏)(7)
q𝑡→𝑠(t)=q𝑠→𝑡(t)𝑇(8)
where𝑠,𝑡are encoded features of skeleton and text. 𝑎=Í𝑖−1
𝑏=1𝑚𝑏+𝑗.
𝑗represents the index of the 𝑗-th text feature within the set of 𝑚text
features that correspond to the 𝑠𝑖keypoint action feature. 𝑠𝑖𝑚(𝑠,𝑡)
is the cosine similarity, 𝜏is the temperature parameter and 𝐵is the
batch size. Unlike image-text pairs in CLI [ 28], which are one-to-
one mappings, in our setting, there could be more than one positive
matching and actions of different categories forming negative pairs.
Given multiple candidate text features for each sign language
sample, we establish a probabilistic match where at least one text
feature𝑡corresponds to each skeleton feature 𝑠. The true corre-
spondence distribution p∈R(𝐵,𝑀)is defined as:
p𝑖=Imatch(𝑠,𝑡𝑖)Í𝑀
𝑐=1Imatch(𝑠,𝑡𝑐)(9)
where the indicator function Imatch(𝑠,𝑡𝑖)equals 1 when 𝑠and𝑡𝑖share
the same sign label, and 0 otherwise. This formulation explicitly
captures the multi-positive relationships between skeleton features
and their corresponding text descriptions.
The contrastive learning objective employs KL divergence to
align the predicted and true distributions:
Lcon(𝑠,𝑡)=1
2Es,t∼Dh
KL(𝑞𝑠→𝑡(s),p)+KL(𝑞𝑡→𝑠(t),p𝑇)i
(10)
where𝐷is the entire dataset.
This symmetric loss function generalizes standard single-positive
contrastive learning [27], where preduces to a one-hot vector. Al-
though related to [ 31], our key distinction is the explicit treatment
of all texts of the same label as matched positives. Optimization
brings together partial descriptions of identical signs while separat-
ing them from different signs, enhancing feature discriminability.
3.3.2 Hierarchical Part Contrastive Learning. Considering the prior
of human body parts, the skeleton can be divided into multiple
groups. We illustrate the overall architecture in Figure 2. We apply
contrastive loss on different part features as well as global feature,
and propose a multipart contrastive loss. The part feature could
be obtained with part pooling, where joint features within the
same group are aggregated to generate a part representation. More
specifically, we choose the features before the final classification
layer for part feature pooling. Our part partition strategy is five-part
partition. The body is divided into five groups: left hand, right hand,

MM ’25, October 27–30, 2025, Dublin, Ireland Siyu Liang, et al.
face, mouth, and body. We then engage in contrastive learning by
comparing the sign-relevant partial text features, obtained from
section 3.2, with their corresponding body parts.
The loss function of multipart contrastive loss can be represented
as follows:
L𝑚𝑢𝑙𝑡𝑖
𝑐𝑜𝑛 =1
𝑁 
L𝑐𝑜𝑛(S𝑔,T𝑔)+L𝑐𝑜𝑛(S𝑔,T𝑠)+𝑃∑︁
𝑖=1L𝑐𝑜𝑛(S𝑝𝑖,T𝑝𝑖)!
(11)
where𝑁is the number of terms used to calculate the contrastive
loss. The hierarchical approach offers distinct advantages by inher-
ently respecting the compositional nature of sign language while
enabling robust learning from partial observations. Through ex-
plicit modeling of part-whole relationships, it achieves superior
generalization. As demonstrated in our experiments, this multi-
granularity representation significantly outperforms global-only
contrastive approaches while maintaining computational efficiency.
3.4 Composite Training Objective
Having introduced the individual components, we now formalize
the complete optimization objective that jointly trains the skele-
ton encoder and text alignment modules. The composite training
objective combines classification and contrastive losses:
L𝑡𝑜𝑡𝑎𝑙=L𝑐𝑙𝑠(S𝑔)+𝛼L𝑚𝑢𝑙𝑡𝑖
𝑐𝑜𝑛(S,T) (12)
whereL𝑐𝑙𝑠denotes the standard cross-entropy classification loss,
L𝑚𝑢𝑙𝑡𝑖𝑐𝑜𝑛 represents our multi-positive contrastive loss, and 𝛼is a
fixed weighting parameter balancing the two objectives. The weight-
ing parameter 𝛼controls the relative importance of semantic align-
ment versus classification accuracy, which we empirically set to 0.5
based on validation performance.
During training, the skeleton encoder processes input poses to
generate S𝑔through the global average pooling of all joint nodes,
while S𝑝is obtained by average pooling predefined groups of re-
lated joints. Both features are projected to match the text feature
dimension via separate fully connected layers. The text descriptions
generated by the GSP are encoded in fixed-dimensional represen-
tations Tusing the text encoder. At inference time, the method
utilizes only the global features of the skeleton encoder S𝑔for the
final prediction, ensuring no additional computational overhead
compared to conventional skeleton-based approaches. During in-
ference, the text encoder and part-specific branches are discarded,
reducing the computational graph to a single-stream skeleton en-
coder matching the efficiency of conventional approaches while
retaining the benefits of multimodal training.
4 Experiments
4.1 Experimental Setup
4.1.1 Datasets. We conduct comprehensive evaluations on two
large-scale sign language recognition datasets.
SLR-500 [11] is a Chinese sign language dataset containing 500
vocabulary items performed by 50 native signers under controlled
laboratory conditions. The dataset is divided into 87,500 training
videos (35 signers) and 37,500 test videos (15 signers) for signer-
independent evaluation. It provides comprehensive coverage of
basic Chinese signs with natural variations in signing styles.AUTSL [29] features 226 Turkish sign language gestures col-
lected from 43 signers, with 28,142 training and 3,742 test samples.
The dataset presents unique challenges due to significant inter-
signer variations and diverse recording conditions. Its medium-
scale vocabulary and realistic signing variations make it particularly
valuable for robustness evaluation.
4.1.2 Implementation details. We employ HR-Net [ 30] to extract
87 semantically significant keypoints per frame, which our ablation
studies identified as optimal for the representation of sign language.
These keypoints are processed by a CTR-GCN [ 3] backbone with
multiscale temporal convolution, preserving both spatial and tem-
poral dynamics. The CLIP text encoder [ 28] processes multiple
description variants, including global descriptions 𝑇𝑔, synonym
variations𝑇𝑠, and part-specific descriptions 𝑇𝑝, using a contrastive
loss temperature parameter 𝜏=0.1throughout all experiments.
Training protocols vary by dataset: for SLR-500 we employ 110
training epochs with batch size 120, implementing a 5-epoch linear
warm-up phase followed by an initial learning rate 0.06 (gradually
annealed by ×0.1 at epochs 90 and 100) and weight decay 5e-4.
The AUTSL dataset follows similar 110-epoch training, but with
batch size 80, initial learning rate 0.04, and weight decay 4e-4 while
maintaining identical reduction scheduling.
Our method consists of generated content in authoritative sign
language resources, including three official Chinese dictionaries
[6,7,26] and the Turkish National Sign Language Dictionary [ 18].
For text generation, we utilize Moonshot1AI models (v1-8K/v1-
32K) to produce synonym variants, expert-verified descriptions,
and part-specific texts. All implementations use PyTorch running
on NVIDIA A100-PCIE-40GB GPU hardware.
4.2 Comparison with State-of-the-Art Methods
In this section, we compare our method with the existing state-of-
the-art (SOTA) methods using the SLR-500 dataset. Additionally,
we also conduct performance comparisons using the large-scale
AUTSL sign language dataset.
4.2.1 Performance Comparison on SLR-500 Dataset. Current SOTA
methods exhibit several limitations that our approach addresses.
While SignBERT effectively models hand gestures as visual tokens
and BEST successfully applies BERT-style pre-training to triplet
units, these methods predominantly focus on manual features, po-
tentially neglecting crucial non-manual elements like facial ex-
pressions and body postures. Similarly, MASA’s motion-aware au-
toencoder, though powerful, faces an information bottleneck from
single-positive contrastive learning.
In contrast, our method demonstrates significant advantages
in several aspects. First, our method benefits from the action de-
scriptions of various body parts related to sign language in the
text features, emphasizing various action features of the human
body related to sign language. By incorporating these text features,
our model can more comprehensively understand sign language
actions, including not only hand gestures, but also other impor-
tant information such as facial expressions and body postures. This
helps the model capture subtle differences in sign language more ac-
curately, achieving performance comparable to or even better than
1https://platform.moonshot.cn/

Generative Sign-description Prompts with Multi-positive Contrastive Learning for Sign Language Recognition MM ’25, October 27–30, 2025, Dublin, Ireland
Table 2: Performance comparison on SLR-500 dataset.
Method Accuracy(%)
ST-GCN [34] 90.0
SignBERT [10] 94.5
BEST [39] 95.4
MASA [38] 96.3
SSRL [40] 96.9
Ours (joint\joint_motion) 96.01\95.28
Ours (bone\bone_motion) 94.37\94.19
Ours (4 streams fusion) 97.1
Table 3: Performance comparison on AUTSL dataset (Top-1
and Top-5 accuracy in %).
Method Top-1 Top-5
SL-TSSI-DenseNet [19] 93.13 –
SSTCN [14] 93.37 –
SAM-SLR-V1 [14] 95.45 99.25
AM-GCN-A [24] 96.27 99.48
SAM-SLR-V2 [15] 96.47 99.76
TMS-Net [4] 96.62 99.71
SML [5] 96.85 99.79
Ours 97.07 99.89
SOTA methods on the SLR-500 dataset. Furthermore, our method
performs well in feature representation at both the joint and bone
levels, indicating that the model can effectively utilize feature infor-
mation at different levels. Particularly in the 4s setting, our method
achieved an accuracy of 97.1%, demonstrating its powerful capabil-
ity to handle more complex sign language action sequences.
4.2.2 Performance Comparison on AUTSL Dataset. To verify the
generalization of our method, we also conducted experiments on
the AUTSL dataset. The AUTSL dataset is known for its diversity
and complexity, which poses greater challenges for SLR models. As
shown in Table 3, our method also demonstrated excellent perfor-
mance on the AUTSL dataset, achieving the highest Top-1 accuracy
of 97.07% and Top-5 accuracy of 99.89%, representing a significant
improvement over previous methods.
Performance improvement can be attributed to the multi-positive
contrastive learning mechanism, which provides a boost in accuracy
of 1.36% by capturing inter-sign variations. Besides, our multimodal
fusion strategy outperforms existing methods through comprehen-
sive integration of manual and non-manual features. These results
confirm our method’s strong generalization across different sign
languages and datasets, maintaining consistent performance ad-
vantages while handling AUTSL’s inherent variability challenges.
4.3 Ablation Study
4.3.1 Combining Sign Language with LLMs. Our initial experiments
revealed limitations when directly applying general purpose LLMs
for the generation of sign language descriptions. The generated
output contained substantial hallucinations, specifically fabricatedTable 4: Effectiveness of GLM, KB, and optimized prompts in
sign language recognition. VE: Visual Encoder.
VE LLM KB Optimized Prompt Accuracy (%)
✓ – – – 93.85
✓ ✓ – – 93.57
✓ ✓ ✓ – 94.89
✓ ✓ ✓ ✓ 95.25
Table 5: Ablation study of hierarchical multipart contrastive
learning (based on joint data).
VE Synonym Prompt Multipart Accuracy (%)
✓ – – – 93.85
✓ ✓ – – 94.50
✓ – ✓ – 94.93
✓ – – ✓ 95.25
✓ ✓ ✓ – 95.38
✓ ✓ ✓ ✓ 95.81
Table 6: Comparison of the effects of different keypoint com-
binations in sign language recognition.
Method Num. Keypoints Acc (%) Parts +Multipart (%)
all 133 59.22 – –
base 27 93.46 3 93.62 (0.16% ↑)
MASA 49 93.66 3 94.01 (0.35% ↑)
Cosign 76 93.57 5 94.52 (0.95% ↑)
Ours 87 93.85 5 95.25 (1.40%↑)
action descriptions misaligned with actual signs, leading to a 0.27%
accuracy degradation in skeleton-text fusion tasks, as quantified in
Table 4. To address these challenges, RAG is strategically utilized
with domain-specific expert-validated sign language corpora to
ensure description accuracy. Moreover, specialized prompts are
crafted to filter non-action references (e.g., “represents agreement”).
The ablation study demonstrates the contribution of each el-
ement: the base visual encoder achieves 93.85% accuracy, while
the raw LLM outputs degrade performance by -0.28%. The inte-
gration of a knowledge base (KB) not only fully recovers but also
substantially exceeds the baseline by +1.04%. Furthermore, opti-
mized prompts provide an additional improvement of +0.36%. This
systematic evaluation validates the effectiveness of our approach in
addressing LLM hallucinations and enhancing the quality of action
descriptions for sign language recognition.
4.3.2 Multipart Contrastive Learning for Sign Language Recognition
.Our method enhances traditional classification-based SLR through
a novel multipart contrastive learning mechanism. This approach
effectively leverages LLM-generated knowledge (detailed action
descriptions, synonyms, and expert grounded part-specific texts)
to deepen the model’s understanding of sign language semantics
by aligning visual and textual representations.

MM ’25, October 27–30, 2025, Dublin, Ireland Siyu Liang, et al.
The experimental results show a successive improvement in per-
formance with each additional element introduced. The inclusion
of synonym variants contributes an increase of +0.65%, while the
use of optimized prompts adds another +1.08% to the accuracy.
Moreover, incorporating precise part-specific descriptions delivers
the most substantial improvement of +1.40%. When all these ele-
ments are fully integrated into the complete method, it achieves an
impressive accuracy of 95.81%. This validates the effectiveness of
our contrastive learning strategy, which successfully bridges the
gap between visual and textual representation spaces, captures the
nuanced differences in actions, and combines expert knowledge
with the capabilities of language models.
4.3.3 Keypoint Selection Analysis. The selection of keypoint com-
binations significantly impacts the recognition performance in
skeleton-based SLR. Table 6 compares various approaches, demon-
strating that our method achieves superior accuracy (93.85%) with
87 keypoints, outperforming existing combinations (27-76 key-
points) while maintaining computational efficiency.
There are limitations to previous keypoint selection strategies.
SAM-SLR [ 14] significantly improved recognition accuracy by re-
ducing the number of keypoints from 133 to 27. However, these
methods have certain limitations in keypoint selection. SAM-SLR
and MASA [ 38] neglect facial, lip, and other non-hand informa-
tion related to sign language; although CoSign [ 16] collects more
keypoints, it still does not cover all relevant body parts.
Through systematic examination of expert sign language descrip-
tions, we identify 87 keypoints covering both manual (hands) and
non-manual (face, mouth, body) locations commonly used in real-
world signing. When combined with our multipart contrastive learn-
ing, this selection achieves a 1.4% accuracy improvement, which is
significantly higher than the gains of other methods (0.16-0.95%).
The results validate that complete coverage and textual grounding
yield optimal recognition performance.
4.4 Visualized Analysis
To provide qualitative and deeper insights into our method’s behav-
ior, we conduct a comprehensive visual analysis using the SLR-500
trained model. This examination reveals how the integration of
textual descriptions influences both spatial attention patterns and
categorical performance across diverse sign categories.
4.4.1 Visualization of Human Keypoint Attention Weights. In Figure
5, we visualize the attention patterns of our skeleton encoder to an-
alyze the impact of contrastive learning based on multipart descrip-
tions. The attention weights are derived from channel-averaged
feature maps, where joint importance is computed through nor-
malized attention aggregation. Before incorporating text features,
the model exhibits diffuse attention across multiple joints, indicat-
ing weaker semantic alignment. After applying our MC method
with text guidance, the model significantly reduces attention noise,
focusing more precisely on linguistically meaningful regions. For
instance, in Class “Tongue”, the model correctly emphasizes left-
hand and facial joints, while Class “Ear” shows enhanced attention
to relevant inter-joint relationships. The visualization confirms
that our method not only suppresses irrelevant keypoints but also
strengthens the model’s ability to capture sign language semantics.
Ear
 Tongue(a) S keleton Only (b) Skeleton with TextsLawyer
 Ice Cream
Figure 5: Attention Visualization Comparison: Skeleton-
only (Top) baseline showing diffuse attention patterns.
Our method’s refined attention (Bottom) focusing on se-
mantically relevant joints and relationships. Color coding:
pink nodes=joint attention (size indicates importance), blue
links=joint relationships (width/opacity indicate strength).
/uni0000002c/uni00000046/uni00000048/uni00000003/uni00000026/uni00000055/uni00000048/uni00000044/uni00000050/uni00000028/uni0000005b/uni00000053/uni00000048/uni00000055/uni0000004c/uni00000048/uni00000051/uni00000046/uni00000048/uni0000003c/uni00000052/uni00000058/uni00000031/uni00000052/uni00000056/uni00000048 /uni00000028/uni00000044/uni00000055/uni00000026/uni00000052/uni0000004f/uni00000047/uni00000036/uni0000004c/uni00000055/uni00000036/uni00000052/uni00000049/uni00000057/uni00000026/uni0000004b/uni0000004c/uni0000004f/uni00000047/uni0000002a/uni0000004c/uni00000055/uni0000004f
/uni0000002b/uni00000058/uni00000050/uni00000044/uni00000051/uni00000032/uni00000055/uni00000044/uni00000051/uni0000004a/uni00000048/uni00000028/uni00000044/uni00000056/uni0000005c/uni0000002b/uni00000044/uni00000053/uni00000053/uni00000048/uni00000051/uni00000037/uni00000048/uni0000004f/uni00000048/uni00000059/uni0000004c/uni00000056/uni0000004c/uni00000052/uni00000051 /uni00000036/uni00000048/uni00000053/uni00000044/uni00000055/uni00000044/uni00000057/uni00000048/uni00000030/uni00000044/uni00000047/uni00000044/uni00000050/uni00000048/uni00000036/uni00000052/uni00000046/uni0000004c/uni00000048/uni00000057/uni0000005c/uni00000027/uni00000044/uni00000057/uni00000048/uni00000035/uni00000048/uni00000053/uni00000048/uni00000044/uni00000057/uni00000036/uni0000004f/uni00000052/uni0000005a/uni0000002e/uni00000048/uni0000005c/uni00000036/uni0000004f/uni0000004c/uni00000050/uni0000002c
/uni00000024/uni00000046/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000046/uni0000004f/uni00000044/uni00000056/uni00000056/uni00000014/uni00000013
/uni00000018
/uni00000013/uni00000018/uni00000014/uni00000013/uni00000014/uni00000018/uni00000024/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c/uni00000003/uni0000004a/uni00000044/uni0000004c/uni00000051/uni00000003/uni0000000b/uni00000008/uni0000000c
Figure 6: Class-wise Performance Gains
4.4.2 Class-wise Performance Gains. Figure 6 presents a detailed
analysis of the class-wise accuracy improvements of our GSP-MC
method versus the baseline CTR-GCN model on SLR-500. The
visual-text contrastive learning method produces markedly dif-
ferent impacts across sign categories, revealing important patterns
about its operational mechanisms.
The substantial 17.33% gain for “ice cream” stems from its precise
description specifying “bend fingers + press cheek + pinch thumb-
index”, while “experience” improves by a notable 14.67% with ex-
plicit “circular motion near forehead” trajectory details. Conversely,

Generative Sign-description Prompts with Multi-positive Contrastive Learning for Sign Language Recognition MM ’25, October 27–30, 2025, Dublin, Ireland
“thick” declines 12% due to its vague “press down” description lack-
ing body part and trajectory specifics, and “I” drops 9.33% when
generated texts confuse pronouns. Recognition accuracy proves to
be directly dependent on complete descriptions, particularly when
specifying exact body configurations (“bend fingers”), locations
(“cheek”), and motions (“circular”).
These results collectively demonstrate that our approach excels
most for signs with concrete physical referents that permit clear
part-specific decomposition. Performance variations directly corre-
late with description quality. The 4/5 of the classes maintaining or
exceeding baseline performance confirms the overall robustness of
the multipart contrastive learning approach.
5 Conclusion
This paper introduced GSP-MC, an innovative method for sign
language recognition that combines the generative capabilities of
LLM with advanced contrastive learning techniques. The Genera-
tive Sign-description Prompts leverage RAG to produce accurate,
multipart sign descriptions grounded in expert knowledge. The
Multi-positive Contrastive learning effectively aligns visual skele-
ton features with multiple textual representations of each sign,
capturing nuanced variations in sign execution. Comprehensive
evaluations on Chinese and Turkish sign language datasets demon-
strated significant improvements over existing methods, achieving
97.1% accuracy on SLR-500 and 97.07% on AUTSL. The method’s
ability to model fine-grained sign dynamics while maintaining com-
putational efficiency highlights its real-world potential for practical
deployment. Future research directions include improving robust-
ness to imperfect pose estimation and extending the approach to
continuous sign language understanding.
References
[1]Hangbo Bao, Li Dong, Furu Wei, Wenhui Wang, Nan Yang, Xiaodong Liu, Yu
Wang, Jianfeng Gao, Songhao Piao, Ming Zhou, and Hsiao-Wuen Hon. 2020.
UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-
Training. In Proceedings of the 37th International Conference on Machine Learning
(Proceedings of Machine Learning Research, Vol. 119) , Hal Daumé III and Aarti Singh
(Eds.). PMLR, Virtual, 642–652. https://proceedings.mlr.press/v119/bao20a.html
[2]Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020.
A simple framework for contrastive learning of visual representations. In Pro-
ceedings of the 37th International Conference on Machine Learning, ICML 2020,
13-18 July 2020, Virtual Event (Proceedings of Machine Learning Research, Vol. 119) .
PmLR, Online, 1597–1607. http://proceedings.mlr.press/v119/chen20j.html
[3]Yuxin Chen, Ziqi Zhang, Chunfeng Yuan, Bing Li, Ying Deng, and Weiming Hu.
2021. Channel-wise topology refinement graph convolution for skeleton-based
action recognition. In Proceedings of the IEEE/CVF international conference on
computer vision . ICCV, Virtual, 13359–13368. doi:10.1109/ICCV48922.2021.01311
[4]Zhiwen Deng, Yuquan Leng, Junkang Chen, Xiang Yu, Yang Zhang, and Qing Gao.
2024. TMS-Net: A multi-feature multi-stream multi-level information sharing
network for skeleton-based sign language recognition. Neurocomputing 572
(2024), 127194. doi:10.1016/J.NEUCOM.2023.127194
[5]Zhiwen Deng, Yuquan Leng, Jing Hu, Zengrong Lin, Xuerui Li, and Qing Gao.
2024. SML: A Skeleton-based multi-feature learning method for sign language
recognition. Knowledge-Based Systems 301 (2024), 112288. doi:10.1016/J.KNOSYS.
2024.112288
[6]Gu Dingqian, Wei Dan, Wang Chenhua, Gao Hui, Yu Yuanyuan, Heng Miao, Qiu
Bing, and Wu Yongsheng. 2019. Chinese manual alphabet . Technical Report GF
0021–2019. Ministry of Education of the People’s Republic of China, State Lan-
guage Commission, China Disabled Persons’ Federation, Beijing, China. National
Standard.
[7]Gu Dingqian, Wei Dan, Yang Yang, Wang Chenhua, Yu Yuanyuan, Gao Hui, Wu
Yongsheng, Heng Miao, Qiu Bing, Liu Wa, Xu Cong, Wang Yibo, Sun Lianqun,
Sun Wanli, and Wang Jian. 2018. Lexicon of Common Expressions in Chinese
National Sign Language . Technical Report GF 0020–2018. Ministry of Education
of the People’s Republic of China, State Language Commission, China DisabledPersons’ Federation, Beijing, China. National Standard.
[8]Hao-Shu Fang, Jiefeng Li, Hongyang Tang, Chao Xu, Haoyi Zhu, Yuliang Xiu,
Yong-Lu Li, and Cewu Lu. 2022. Alphapose: Whole-body regional multi-person
pose estimation and tracking in real-time. IEEE transactions on pattern analysis
and machine intelligence 45, 6 (2022), 7157–7173. doi:10.1109/TPAMI.2022.3222784
[9]Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross B. Girshick. 2020. Mo-
mentum contrast for unsupervised visual representation learning. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . Computer
Vision Foundation / IEEE, 9729–9738. doi:10.1109/CVPR42600.2020.00975
[10] Hezhen Hu, Weichao Zhao, Wengang Zhou, Yuechen Wang, and Houqiang Li.
2021. c. In Proceedings of the IEEE/CVF international conference on computer vision .
ICCV, Virtual, 11087–11096. doi:10.1109/ICCV48922.2021.01090
[11] Jie Huang, Wengang Zhou, Houqiang Li, and Weiping Li. 2018. Attention-based
3D-CNNs for large-vocabulary sign language recognition. IEEE Transactions on
Circuits and Systems for Video Technology 29, 9 (2018), 2822–2832. doi:10.1109/
TCSVT.2018.2870740
[12] Eui Jun Hwang, Sukmin Cho, Junmyeong Lee, and Jong C Park. 2024. An Efficient
Sign Language Translation Using Spatial Configuration and Motion Dynamics
with LLMs. doi:10.48550/ARXIV.2408.10593
[13] Intel. 2019. Intel®RealSense ™Technology . https://www.intel.com/content/
www/us/en/architecture-and-technology/realsense-overview.html Accessed on
25/02/2025.
[14] Songyao Jiang, Bin Sun, Lichen Wang, Yue Bai, Kunpeng Li, and Yun Fu. 2021.
Sign language recognition via skeleton-aware multi-model ensemble. CoRR
abs/2110.06161 (2021). arXiv:2110.06161 https://arxiv.org/abs/2110.06161
[15] Songyao Jiang, Bin Sun, Lichen Wang, Yue Bai, Kunpeng Li, and Yun Fu. 2021.
Skeleton aware multi-modal sign language recognition. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition . Computer Vision
Foundation / IEEE, 3413–3423. doi:10.1109/CVPRW53098.2021.00380
[16] Peiqi Jiao, Yuecong Min, Yanan Li, Xiaotao Wang, Lei Lei, and Xilin Chen. 2023.
Cosign: Exploring co-occurrence signals in skeleton-based continuous sign lan-
guage recognition. In Proceedings of the IEEE/CVF international conference on
computer vision . IEEE, 20676–20686. doi:10.1109/ICCV51070.2023.01890
[17] Sai Shashank Kalakonda, Shubh Maheshwari, and Ravi Kiran Sarvadevabhatla.
2023. Action-gpt: Leveraging large-scale language models for improved and gen-
eralized action generation. In 2023 IEEE International Conference on Multimedia
and Expo (ICME) . IEEE, 31–36. doi:10.1109/ICME55011.2023.00014
[18] Semih Kavak. 2015. Turkish Sign Language Dictionary . National Education
Ministry.
[19] David Laines, Miguel Gonzalez-Mendoza, Gilberto Ochoa-Ruiz, and Gissella Be-
jarano. 2023. Isolated sign language recognition based on tree structure skeleton
images. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition . 276–284. doi:10.1109/CVPRW59228.2023.00033
[20] Siyu Liang, Yunan Li, Yuanyuan Shi, Huizhou Chen, and Qiguang Miao. 2025.
Integrated multi-local and global dynamic perception structure for sign language
recognition. Pattern Analysis and Applications 28, 2 (2025), 1–14. doi:10.1007/
S10044-024-01403-8
[21] Kezhou Lin, Xiaohan Wang, Linchao Zhu, Bang Zhang, and Yi Yang. 2024. SKIM:
Skeleton-based isolated sign language recognition with part mixing. IEEE Trans-
actions on Multimedia 26 (2024), 4271–4280. doi:10.1109/TMM.2023.3321502
[22] Kang Liu, Zhuoqi Ma, Xiaolu Kang, Yunan Li, Kun Xie, Zhicheng Jiao, and
Qiguang Miao. 2025. Enhanced Contrastive Learning with Multi-view Longi-
tudinal Data for Chest X-ray Report Generation. CoRR abs/2502.20056 (2025).
doi:10.48550/ARXIV.2502.20056 arXiv:2502.20056
[23] Ruyi Liu, Yi Liu, Mengyao Wu, Wentian Xin, Qiguang Miao, Xiangzeng Liu, and
Long Li. 2025. SG-CLR: Semantic representation-guided contrastive learning for
self-supervised skeleton-based action recognition. Pattern Recognition 162 (2025),
111377. doi:10.1016/J.PATCOG.2025.111377
[24] Yuhong Liu, Fei Lu, Xianpeng Cheng, and Ying Yuan. 2024. Asymmetric multi-
branch GCN for skeleton-based sign language recognition. Multimedia Tools and
Applications 83, 30 (2024), 75293–75319. doi:10.1007/S11042-024-18443-1
[25] Microsoft. 2017. Developing with Kinect . https://developer.microsoft.com/en-
us/windows/kinect/develop Accessed on 25/02/2025.
[26] China Association of the Deaf and Hard of Hearing (Eds.). 2018. Chinese Sign
Language (Revised Edition) (Parts 1 & 2) . Huaxia Publishing House.
[27] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning
with contrastive predictive coding. CoRR abs/1807.03748 (2018). arXiv:1807.03748
http://arxiv.org/abs/1807.03748
[28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al .2021. Learning transferable visual models from natural language su-
pervision. In International conference on machine learning . PMLR, 8748–8763.
http://proceedings.mlr.press/v139/radford21a.html
[29] Ozge Mercanoglu Sincan and Hacer Yalim Keles. 2020. Autsl: A large scale multi-
modal turkish sign language dataset and baseline methods. IEEE Access 8 (2020),
181340–181355. doi:10.1109/ACCESS.2020.3028072
[30] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. 2019. Deep high-resolution
representation learning for human pose estimation. In Proceedings of the IEEE/CVF

MM ’25, October 27–30, 2025, Dublin, Ireland Siyu Liang, et al.
conference on computer vision and pattern recognition . 5693–5703. doi:10.1109/
CVPR.2019.00584
[31] Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, and Dilip Krishnan.
2023. Stablerep: Synthetic images from text-to-image models make strong vi-
sual representation learners. Advances in Neural Information Processing Sys-
tems 36 (2023), 48382–48402. http://papers.nips.cc/paper_files/paper/2023/hash/
971f1e59cd956cc094da4e2f78c6ea7c-Abstract-Conference.html
[32] Ryan Wong, Necati Cihan Camgöz, and Richard Bowden. 2024. Sign2GPT: Lever-
aging Large Language Models for Gloss-Free Sign Language Translation. In The
Twelfth International Conference on Learning Representations, ICLR 2024, Vienna,
Austria, May 7-11, 2024 . OpenReview.net. https://openreview.net/forum?id=
LqaEEs3UxU
[33] Wangmeng Xiang, Chao Li, Yuxuan Zhou, Biao Wang, and Lei Zhang. 2023.
Generative Action Description Prompts for Skeleton-based Action Recognition.
InProceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) .
IEEE, 10276–10285. doi:10.1109/ICCV51070.2023.00943
[34] Sijie Yan, Yuanjun Xiong, and Dahua Lin. 2018. Spatial temporal graph convolu-
tional networks for skeleton-based action recognition. In Proceedings of the AAAI
conference on artificial intelligence , Vol. 32. doi:10.1609/aaai.v32i1.12328
[35] Tingbing Yan, Wenzheng Zeng, Yang Xiao, Xingyu Tong, Bo Tan, Zhiwen Fang,
Zhiguo Cao, and Joey Tianyi Zhou. 2024. Crossglg: Llm guides one-shot skeleton-
based 3d action recognition in a cross-level manner. In European Conference on
Computer Vision , Vol. 15078. Springer, 113–131. doi:10.1007/978-3-031-72661-3_7[36] Lin Yuan, Zhen He, Qiang Wang, Leiyang Xu, and Xiang Ma. 2022. Skeleton-
clip: Recognizing skeleton-based human actions with text prompts. In 2022
8th International Conference on Systems and Informatics (ICSAI) . IEEE, 1–6.
doi:10.1109/ICSAI57119.2022.10005459
[37] Huaiwen Zhang, Zihang Guo, Yang Yang, Xin Liu, and De Hu. 2023. C2st:
Cross-modal contextualized sequence transduction for continuous sign language
recognition. In Proceedings of the IEEE/CVF International Conference on Computer
Vision . IEEE, 21053–21062. doi:10.1109/ICCV51070.2023.01925
[38] Weichao Zhao, Hezhen Hu, Wengang Zhou, Yunyao Mao, Min Wang, and
Houqiang Li. 2024. MASA: Motion-aware Masked Autoencoder with Semantic
Alignment for Sign Language Recognition. IEEE Transactions on Circuits and
Systems for Video Technology 34, 11 (2024), 10793–10804. doi:10.1109/TCSVT.
2024.3409728
[39] Weichao Zhao, Hezhen Hu, Wengang Zhou, Jiaxin Shi, and Houqiang Li. 2023.
BEST: BERT pre-training for sign language recognition with coupling tokeniza-
tion. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 37.
3597–3605. doi:10.1609/AAAI.V37I3.25470
[40] Weichao Zhao, Wengang Zhou, Hezhen Hu, Min Wang, and Houqiang Li. 2024.
Self-supervised representation learning with spatial-temporal consistency for
sign language recognition. IEEE Transactions on Image Processing 33 (2024),
4188–4201. doi:10.1109/TIP.2024.3416881
[41] Ronglai Zuo, Fangyun Wei, and Brian Mak. 2023. Natural Language-Assisted
Sign Language Recognition. In 2023 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR) . 14890–14900. doi:10.1109/CVPR52729.2023.01430