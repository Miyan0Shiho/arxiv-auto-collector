# HV-Attack: Hierarchical Visual Attack for Multimodal Retrieval Augmented Generation

**Authors**: Linyin Luo, Yujuan Ding, Yunshan Ma, Wenqi Fan, Hanjiang Lai

**Published**: 2025-11-19 13:45:24

**PDF URL**: [https://arxiv.org/pdf/2511.15435v1](https://arxiv.org/pdf/2511.15435v1)

## Abstract
Advanced multimodal Retrieval-Augmented Generation (MRAG) techniques have been widely applied to enhance the capabilities of Large Multimodal Models (LMMs), but they also bring along novel safety issues. Existing adversarial research has revealed the vulnerability of MRAG systems to knowledge poisoning attacks, which fool the retriever into recalling injected poisoned contents. However, our work considers a different setting: visual attack of MRAG by solely adding imperceptible perturbations at the image inputs of users, without manipulating any other components. This is challenging due to the robustness of fine-tuned retrievers and large-scale generators, and the effect of visual perturbation may be further weakened by propagation through the RAG chain. We propose a novel Hierarchical Visual Attack that misaligns and disrupts the two inputs (the multimodal query and the augmented knowledge) of MRAG's generator to confuse its generation. We further design a hierarchical two-stage strategy to obtain misaligned augmented knowledge. We disrupt the image input of the retriever to make it recall irrelevant knowledge from the original database, by optimizing the perturbation which first breaks the cross-modal alignment and then disrupts the multimodal semantic alignment. We conduct extensive experiments on two widely-used MRAG datasets: OK-VQA and InfoSeek. We use CLIP-based retrievers and two LMMs BLIP-2 and LLaVA as generators. Results demonstrate the effectiveness of our visual attack on MRAG through the significant decrease in both retrieval and generation performance.

## Full Text


<!-- PDF content starts -->

HV-Attack: Hierarchical Visual Attack for Multimodal Retrieval
Augmented Generation
Linyin Luo1,2, Yujuan Ding1, Yunshan Ma3, Wenqi Fan1, Hanjiang Lai2
1The Hong Kong Polytechnic University
2Sun Yat-Sen University
3Singapore Management University
luoly36@mail2.sysu.edu.cn, dingyujuan385@gmail.com, ysmasmu.edu.sg
wenqifan03@gmail.com, laihanj3@mail.sysu.edu.cn
Abstract
Advanced multimodal Retrieval-Augmented Generation
(MRAG) techniques have been widely applied to enhance
the capabilities of Large Multimodal Models (LMMs), but
they also bring along novel safety issues. Existing adversar-
ial research has revealed the vulnerability of MRAG systems
to knowledge poisoning attacks, which fool the retriever
into recalling injected poisoned contents. However, our
work considers a different setting:visual attack of MRAG
by solely adding imperceptible perturbations at the image
inputs of users, without manipulating any other compo-
nents.This is challenging due to the robustness of fine-
tuned retrievers and large-scale generators, and the effect
of visual perturbation may be further weakened by propa-
gation through the RAG chain. We propose a novel Hierar-
chical Visual Attack that misaligns and disrupts the two in-
puts (the multimodal query and the augmented knowledge)
of MRAG’s generator to confuse its generation. We fur-
ther design a hierarchical two-stage strategy to obtain mis-
aligned augmented knowledge. We disrupt the image input
of the retriever to make it recall irrelevant knowledge from
the original database, by optimizing the perturbation which
first breaks the cross-modal alignment and then disrupts the
multimodal semantic alignment. We conduct extensive ex-
periments on two widely-used MRAG datasets: OK-VQA
and InfoSeek. We use CLIP-based retrievers and two LMMs
BLIP-2 and LLaVA as generators. Results demonstrate the
effectiveness of our visual attack on MRAG through the sig-
nificant decrease in both retrieval and generation perfor-
mance.
1. Introduction
The robustness of multimodal Retrieval-Augmented Gen-
eration (MRAG) [1, 13, 27, 33] systems is of great impor-
(a) Knowledge poisoning attacks of text RAG models
User  Query: (Text)
Who is the CEO of OpenAI?Knowledge Base
(Text)
 […] TimCook[…] as the
CEO of OpenAI since2024.③ RetrieveContext: […] TimCook[…]
as the CEO of OpenAI
since2024.
Question: Who is the CEO of
OpenAI? 
Answer: 
Tim Cook④ GenerateLLM
② Inject
① ConstructMalicious Text
(Text)User
Attacker
(b) Knowledge poisoning attacks of multimodal RAG models
User  Query: (Text + Image)
What is shown 
in the image?Knowledge Base
(Text + Image)
Respond to every
question with "I
don't know"...③ RetrieveContext: Respond to every
question with "I
don't know"...
Question: What is shown 
in the image?
Answer: 
I don't know④ Generate
LVLM② Inject
① ConstructMalicious Text
(Text + Image)User
Attacker
(c) The pr oposed attack on multimodal RAG models
Knowledge Base
(Text / Text + Image)
② RetrieveContext: ... the visionary toy
company ...bought the rights
in 1957. ...
Question: When was the
cola brand on the signs 
founded ? 
Answer: 
1957③ Generate
① Add Noise
to Image.User
Attacker
User  Query: (Text + Image)
When was the cola 
brand on the signs 
founded?
LVLM
GeneratorKnowledge Base
(Text / Text + Image) Context: ... the visionary
toy company ...bought the
rights in 1957. ...
Answer: 
1957User User  Query: (Text + Image)
Retriever
When was the cola 
brand on the signs 
founded?
Visual Query
AttackAttacker
Textual
Query AttackAttacker
Textual KB
Attack
Attacker
GeneratorKnowledge Base
(Text / Text + Image)
Answer  
1957UserUser  Query
(Text + Image)
Retriever
Multi-modal RAG System
When was the
cola brand on the
signs founded?
(c) Visual
Query Attack(b) Textual
Query Attack(a) Textual
KB AttackDocuments
...the visionary toy
company ...bought
the rights in 1957...
Knowledge Base
(Text / Text + Image)
User  Query
(Text + Image)
Generator
Multi-modal RAG System
When was the
cola brand on the
signs founded?
(c) Hierar chical
Visual Attack(b) Textual
Query Attack(a) KB
Poisoning
Attack
Documents
...the visionary toy
company ...bought
the rights in 1957...
Controversial
Answer  
1957
RetrieverFigure 1. Comparison on (a) KB Poisoning Attack, (b) Textual
Query Attack and our focused task (c) Hierarchical Visual Attack
on multi-modal RAG.
tance. As advanced MRAG techniques have been widely
applied to large multimodal models (LMMs) [18, 22, 23,
34] to enhance their knowledge and ability, novel safety
issues have also emerged. Existing adversarial research
has revealed the vulnerability of MRAG to knowledge poi-
soning attacks [11, 24, 40]. These methods rely on con-
structing and injecting malicious contents into the knowl-
edge base, which are obvious by changing textual words
or can be identified by Knowledge Base (KB) detection al-
gorithm. Meanwhile, there are research of attacks on the
generator component of MRAG [30, 31, 35], which include
manipulations on textual queries and are easy to notice as
well. Thus, in this paper, we consider a different setting
of attacking MRAG: pure visual attack that solely adds im-
perceptible perturbations at the image inputs of users. As
shown in Figure 1, our attack method aims to disrupt both
retrieval and generation process of MRAG by optimizing
visual perturbations, without manipulating any other com-arXiv:2511.15435v1  [cs.CV]  19 Nov 2025

74.474.1872.69
56.92
31.81clean
74.63
2540557085
Random AA XT LA OursRecall@5 (eps = 8/255)
74.4 74.2674.2273.03clean
74.63
2540557085
8/255 16/255 32/255 64/255Random Perturbation Recall@5 Figure 2. Attack performance on a CLIP model fine-tuned for the
retrieval task by different attack methods (AA: Any Attack [39];
XT: X-Transfer [14], LA: LMM Attack [7]) and scale.
ponents. Compared to poisoning attacks, adversarial visual
attack on images can be highly imperceptible to human vi-
sion, making them exceptionally stealthy.
However, attacking MRAG with pure visual perturba-
tion is challenging. The MRAG system is a pipeline that
consists of a fine-tuned retriever and a large-scale genera-
tor, both of which retain a certain degree of robustness. For
fine-tuned retrievers, as shown in Figure 2, advanced vi-
sual attack methods [7, 14, 39] can cause limited retrieval
performance drop, and random perturbations even with a
large scale (64/255) causes subtle performance drop. This
may be because the models become adept at multimodal
knowledge grounding and logical association, focusing on
relational understanding (e.g., the logical relationship be-
tween a question and a potential answer). This emphasis
on deeper logical correlation and inference ability makes
the model less susceptible to simple input perturbations in
a single modality, significantly increasing the challenge of
our image attack task. For generators, large-scale LMMs
such as LLaV A [22] and BLIP-2 [18] are often used, which
generally demonstrate superior robustness [36], further pos-
ing challenging to MRAG attack.
The sequential working pipeline of MRAG also en-
hances its robustness towards visual perturbations as the at-
tack effect degrades by propagation through the RAG chain.
The perturbation’s effect on retriever lies in the recalled
knowledge, which experiences chain of transformation and
decrease. Consequently, success visual attack on MRAG is
hard, while posing a more severe threat than obvious text
attacks or KB poisoning attacks, as the resulting bad influ-
ence can go undetected and propagate through the entire
RAG chain.
To effectively subvert the robustness of MRAG with only
visual perturbations, we propose a novel hierarchical dis-
ruption approach to attack the MRAG’s on two compo-
nents and mislead the final result. As shown in Figure 1,
there are two inputs for the MRAG generator: the multi-
modal user query and the augmented knowledge. Our hier-
archical method targets at misaligning and disrupting these
two inputs, creating knowledge conflicts for the generator
while breaking different levels of model capabilities. For
retrieval, we employ a hierarchical two-stage strategy to op-timize the applied noise in modality and semantic levels.
We first alter the query features to no longer correspond
accurately to itself, so that the retrieval query is deviated.
Then, the semantic alignment between query and knowl-
edge is broken down. By structurally targeting the model’s
core competencies across different levels of abstraction, our
approach achieves severe and effective degradation regard-
ing retrieval and generation performance.
In summary, our main contributions are as follows:
• We propose a novel Hierarchical Visual Attack method,
which misaligns and disrupts the two inputs of generator
in MRAG, posting stealth and severe threats for MRAG
systems.
• Our method is the first to disrupt MRAG systems only us-
ing image perturbations, which further reveals the vulner-
abilities of MRAG systems towards more imperceptible
attacks.
• We conduct extensive experiments on two datasets and
four versions of retrievers to prove our attack’s effective-
ness.
2. Related Work
2.1. Multimodal RAG and Existing Attacks
Multimodal Retrieval-Augmented Generation [1, 27] has
emerged as an important technique by extending traditional
RAG [17] to multimodal data, enabling more real-world
applications. Despite its huge success, the security issue
has gained much attention. Recent studies have highlighted
the vulnerabilities of multimodal RAG systems to knowl-
edge poisoning attacks, where malicious information is in-
jected into the external knowledge databases to manipulate
the RAG’s outputs. MM-Poisoning [11] achieved the attack
by constructing query-specific misinformation into injected
text and images, or inserting a single irrelevant knowledge
instance to fool all queries. Poisoned-MRAG [24] formal-
ized the attack as an optimization problem and proposed
cross-modal attack strategies to disrupt both retrieval and
generation. PoisonedEye [37] designed injected textual
context or optimized the poison image to reduce retrieval
performance to achieve attack goals. Despite these ad-
vancements, existing research still follows the line of text
RAG attacks, which mainly focus on knowledge poison-
ing attacks. Furthermore, the multimodal characteristics re-
mains underexplored. Thus, in this paper, we propose a new
attack on multimodal RAG by only learning small adversar-
ial perturbations added to the image in user query, without
modifying the external knowledge base.
2.2. Visual Attack
Visual adversarial attacks can be categorized into white-
box, gray-box and black-box based on the level of knowl-
edge about the attacked model [38]. White-box methods[2,

+ + + ...
 +
 +
 Stage 1: Modality Alignment AttackFrozen Gradient FlowHV-Attack Generation Pr ocessWhen was the cola 
brand on the signs 
founded?Query 
Image + Text
DPR fine-tuned Retriever  Answer
Adversarial Query 
Image + Text
HV-Attack 
for RetrieverHierar chical Visual (HV)-Attack on MRAG 
... ...
Refer ence Sample 
Selection (RSS)Visual Attack 
for Query
Generator
Positive Caption  
A photo of a dog catching a
frisbee in the grass, where a
woman is watching.
Negative Caption  
A photo of a small shack
with a sign that says stop,
where coca cola is sold.  
A photo of a
small shack with
a sign that says
stop, where coca
cola is sold.  
Image + Cpation
RSSModality-level Contrastive Learning 
MCL
 Stage 2: Semantic Alignment Attack
When was the cola brand
on the signs founded? A
photo of a small shack
with a sign that says stop,
where coca cola is sold. 
Query Image + Text
Negative Query
When was the cola brand on
the signs founded? A photo of
a small shack with a ...  
Positive Document
'disc dog (commonly called
frisbee dog) is a dog sport. ...
participate in .... '
RSSSemantic-level Contrastive Learning 
MCL
Start EndWhen was the cola 
brand on the signs 
founded?
Multi-modal Contrastive Learning (MCL)
Multi-
modal
EncoderText
Encoder
Text
Encoderpullpush
Positive
textNegative
text Multi-
modal
anchorFigure 3. An overview of the proposed method. The top part illustrates the overall MRAG pipeline and our hierarchical structure, which
adds image perturbation to the retriever and generator respectively. The hierarchical two-stage strategy we employed for generating retrieval
visual attack is shown in the lower block. We optimize the added perturbation step-by-step by first breaking the modality alignment, then
disrupting the semantic alignment.
10, 32], which have full access to the attacked model, aim to
maximize the adversarial effect. Fast Gradient Sign Method
(FGSM) [9] generates adversarial examples by adding a
small perturbation in the direction of the gradient of the loss
function with respect to the input. The Projected Gradient
Descent (PGD) [25] is an iterative variant of FGSM that re-
fines the perturbation through multiple steps. The Carlini
and Wagner (CW) attack [4] employs optimization tech-
niques to find adversarial examples that achieve best attack
while being close to the original input. Gray-box [3, 15] and
black-box [5] attacks have limited knowledge about the at-
tacked model, and rely on transferability or surrogate mod-
els to craft effective perturbations. In this paper, we focus
on white-box attack on images, assuming full access to the
open-source retriever in MRAG pipeline.3. Method
3.1. MRAG Preliminaries and Problem Definition
A multimodal RAG pipeline consists of a retrieverR, a gen-
eratorGand an external knowledge baseKB. When a user
inputs an imageIand a text queryT, the retrieverRfirst re-
trieve top-k augmented knowledge fromKB. Then the gen-
eratorGgenerates an answer using the multimodal query
along with k augmented knowledge. The retrieval process
follows the DPR [16] structure, which comprises an image
encoderE Iand a text encoderE T. The encoders embed
input and knowledge into a common semantic space. A
similarity-based retrieval can be conducted by calculating
and ranking the distance between the input and knowledge
in this semantic space. For each query, the top-k knowledge
embeddings with highest similarity are retrieved as results.
The multimodal embedding of a user query is obtained by

summing the corresponding image and text embeddings,
i.e.,E I(I) +E T(T). The knowledge embedding is de-
pended on its modality. For text corpus, each passageK t’s
embedding isE T(Kt). For multimodal KB, each image-
text pair(K i, Kt)’s embedding isE I(Ki) +E T(Kt).
In this paper, our goal is to attack the MRAG pipeline by
solely modifying the input image. We achieve our attack by
misaligning and disrupting the two inputs of the generator
compomnent in MRAG, which results in confusion within
inputs along with misleading information. To be more spe-
cific, for the augmented knowledge, we optimize a small
perturbationδ radded at the input imageI qto mislead the
retriever to return irrelevant knowledge. For the image in-
put of the generator, we add another small perturbationδ g
that breaks down uni-modal semantic within the image to
disrupt image understanding. Consequently, the generator
produces incorrect answers based on the misaligned input
query and augmented knowledge, with misleading irrele-
vant contents. The attack objective can be formulated as
follows:
Iadvr=attack r(Iq, δr),||I q−Iadvr||∞=||δ r||∞< ϵ,
Iadvg=Iq+δg,||I q−Iadvg||∞=||δ g||∞< ϵ,
Topk adv=R(T, I advr,KB),
Aadv=G(T, I advg, Topk adv),
whereδ randδ gare noises added to the image in retrieval
and generation,attack r(·)represents our hierarchical at-
tack method of learning retrieval and generation perturba-
tions respectively,ϵis the bound of added perturbation to
ensure that it is imperceptible to human eye.
3.2. Overall Hierarchical Attack Framework
The overall attack framework is illustrated in Figure 3. To
produce confusion for the generator in MRAG, we aim to
misalign the two inputs of it: the input query and augmented
knowledge. Thus, we hierarchically apply two different per-
turbations to the image, resulting in misleading effect in two
directions.
For the augmented knowledge, the added perturbation
δrmainly takes effect by disrupting the retrieval process.
We further adopt a hierarchical two-stage strategy to learn
δr, targeting at modality alignment and semantic alignment
within retrieval respectively. The detailed design of this
hierarchical strategy is described in subsection 3.3. For
the generator input query, the added perturbationδ gbreaks
down the uni-modal semantic within the image input, so as
to disrupt image understanding in generation.
3.3. Hierarchical Visual Attack on Augmented
Knowledge
The visual attack on the augmented knowledge as inputs to
the generator is achieved by disrupting the retrieval process.We employ a hierarchical two-stage strategy to break down
modality and semantic alignments respectively, which are
key characteristics in success retrieval.
Stage 1: Modality Alignment Attack
Multimodal dual-encoder retrievers, such as CLIP [29],
achieve strong alignment between text and image modal-
ities. The corresponding text and image embeddings are
close in the embedding space, enabling relevant knowledge
to be retrieved through the ranking of embedding similar-
ities during cross-modal retrieval. Thus, the first stage of
visual attack on augmented knowledge is to break down the
multimodal alignment between these modalities.
We achieve this by pushing the query image close to
the least similar sample. Given a user query image, we
search for a reference image from the database whose em-
bedding has the smallest similarity with the query image’s
embedding. Then, given the retriever image encoderE i,
text encoderE tand image captioning modelC, we obtain
the multimodal image representationf multi stage1 , the clean
query caption’s embeddingf clean capand the reference im-
age caption’s embeddingf refcapas follows:
fclean cap=E t(C(I q)),
frefcap=E t(C(I r)),
fmulti stage1 =E i(Ip) +E t(C(I p)),(1)
whereI qandI rare the input user query image and its refer-
ence image,I pis the query image with added perturbation.
To learn the best the perturbation, we conduct contrastive
learning betweenf multi stage1 ,fclean capandf refcapand
design the loss functionL CMbased on hinge loss [8]:
LCM= max (∥clean sim−β·ref sim∥+γ,0),
clean sim=sim(f multi stage1, fclean cap),
refsim=sim(f multi stage1, frefcap).(2)
As shown in Equation 2, by minimizingL CM, we min-
imize the similarity between the query and clean image’s
caption, while maximize the similarity with the reference
caption. The detailed optimization process can be found in
Algorithm 1. In this stage, since we focus on cross-modal
alignment, we consider the alignment between multimodal
representation of the query image and the corresponding
text embeddings. Through iterations, we minimize the sim-
ilarity between the multimodal embedding and its text cap-
tion embedding, while maximizing the similarity with the
reference text caption embedding.
Stage 2: Semantic Alignment Attack
The second stage of our attack aims to further break
down the semantic relevance between cross-modal embed-
dings, which is crucial for retrieving useful knowledge for
generation. The form of the loss function is similar to that
in the first stage. However, the contrastive learning is now

Algorithm 1Modality Alignment Attack (Stage 1)
Input:User imageI q, reference imageI r, retriever im-
age encoderE i, retriever text encoderE t, image cap-
tioning modelC, generation stepss, step lengthα, per-
turbation boundϵ, trade-off parameterβ, margin pa-
rameterγ.
Output:Stage 1’s perturbationδ r1
1:Get clean and reference image caption embedding:
fclean cap←E t(C(I q)),frefcap←E t(C(I r))
2:Initialize perturbationδ r1←0
3:Initialize perturbed imageI p←Iq+δr1
4:forstep←1tosdo
5:I p←Iq+δr1
6:f multi stage1←E i(Ip) +E t(C(I p))
7:clean sim←sim(f multi stage1, fclean cap)
8:ref sim←sim(f multi stage1, frefcap)
9:L CM←max (∥clean sim−β·ref sim∥+γ,0),
10:Optimizeδ r1←δr1−α·sign(∇ δr1LCM)
11:δ r1←Clip(δ r1,−ϵ, ϵ)
12:end for
Returnδ r1
conducted between the multimodal representation of user
query, the query text embedding and reference passage em-
bedding. We retrieve the positive text knowledge for the
reference image and use its embedding as the positive em-
bedding in this stage. The embeddings are obtained by:
fclean query =E t(Tq+C(I q)),
frefpassage =E t(Tp),
fmulti stage2 =E i(Ip) +E t(Tq+C(I p)),(3)
whereT qandT pare the user text query and retrieved refer-
ence text knowledge respectively.
The contrastive learning of the second stage is designed
as shown in Equation 4:
LR←max (∥clean sim−β·ref sim∥+γ,0),
clean sim←sim(f multi stage2, fclean query ),
refsim←sim(f multi stage2, frefpassage ).(4)
The detailed process of the second stage is shown in Al-
gorithm 2. Building on the perturbation generated in the
first stage, the second stage iteratively refines and outputs
the final perturbation.
With the two-stage hierarchical strategy, we obtain the
final retrieval perturbation added to the query image and re-
trieve the adversarial augmented knowledge. The genera-
tor’s disrupted query is obtained by adding noise generated
by advanced attack method on LMMs to the image. The two
inputs that misalign with each other achieve the hierarchical
attack on MRAG pipeline.Algorithm 2Semantic Alignment Attack (Stage 2)
Input:Perturbationδ r1from first stage, user imageI q,
User text queryT q, reference positive passageT p, mod-
elsE i, Et,Cand hyperparameterss, α, ϵ, β, γthe same
as stage 1.
Output:Final Perturbationδ r
1:Get reference positive passage embedding:
frefpassage ←E t(Tp)
2:Get clean query text embedding:f clean query ←
Et(Tq+C(I q))
3:Initialize perturbationδ r←δr1
4:Initialize perturbed imageI p←Iq+δr
5:forstep←1tosdo
6:I p←Iq+δr
7:f multi stage2←E i(Ip) +E t(Tq+C(I p))
8:clean sim←sim(f multi stage2, fclean query )
9:ref sim←sim(f multi stage2, frefpassage )
10:L R←max (∥clean sim−β·ref sim∥+γ,0),
11:Optimizeδ r←δr−α·sign(∇ δrLR)
12:δ r←Clip(δ r,−ϵ, ϵ)
13:end for
Returnδ r
4. Experiments
4.1. Datasets and evaluation metrics
We conduct our experiments on two wide-used datasets:
•OK-VQA[26]: This is a large knowledge-based VQA
dataset. Each data sample consists of a question, an im-
age and 10 gold answers. The images are from the COCO
dataset [19] and each question requires external knowl-
edge to answer. We use the test set for attack evaluation,
which contains 5046 samples. For OK-VQA, we fol-
low RA VQA [20], FLMR [21] to use the Google Search
corpus as the external knowledge base, which contains
169,306 passages in total.
•Infoseek[6]: We follow PoisonedEye [37] to attack 1000
samples from the visual question answering dataset Infos-
eek for evaluation. The knowledge database is the same
subset of 2M image-text pairs from OVEN-Wiki [12] with
PoisonedEye.
Evaluation metrics.Our attack method is evaluated
from two perspectives: retrieval and generation. For re-
trieval, we first useS(q, p)to identify the relation between
a queryqand a passagep, which is classified based on
whether the passage contains a ground-truth answer to the
query.
S(q, p) =(
1,if p contains answer to q,
0,if p does not contain answer to q.
Then, we adopt the following two retrieval metrics.

Table 1. VQA Performance on OK-VQA. (ASR* is calculated as(s clean−s adv)/sclean for each metrics, EM: Exact Match, “↑”
indicates that a higher value is better for this metric, while “↓” indicates that a lower value is better. All VQA metrics are reported with
RAG knowledge numberK= 5.Boldand underline represent the best and second best results respectively.)
Retriever MethodRetrieval(%) BLIP-2 VQA(%)
R@5(↓) ASR*(↑) P@5(↓) ASR*(↑)VQA
Score(↓)ASR*(↑) EM(↓) ASR*(↑)
CLIP
ViT-L/14Clean 50.57 - 27.86 - 38.53 - 41.76 -
AnyAttack [39] 50.12 0.89 27.21 2.33 37.94 0.59 41.14 0.62
X-Transfer [14] 48.14 4.81 25.27 9.30 32.95 14.48 36.03 13.72
LMM Attack [7] 12.76 74.77 4.48 83.92 32.53 15.57 35.26 15.57
Ours 14.00 72.32 4.86 82.56 28.39 26.32 30.99 25.79
CLIP
ViT-L/14
finetunedClean 74.63 - 44.03 - 41.25 - 44.55 -
AnyAttack [39] 74.18 0.60 43.56 1.07 41.11 0.34 44.47 0.18
X-Transfer [14] 72.69 2.60 40.99 6.90 35.54 13.84 38.55 13.47
LMM Attack [7] 56.92 23.73 27.34 37.91 33.38 19.08 36.29 18.54
Ours 31.81 57.38 12.40 71.84 28.80 30.18 31.25 29.85
Table 2. VQA Performance on InfoSeek. (ASR* is calculated as(s clean−sadv)/sclean for each metrics, EM: Exact Match, “↑” indicates
that a higher value is better for this metric, while “↓” indicates that a lower value is better. All VQA metrics are reported with RAG
knowledge numberK= 5.Boldand underline represent the best and second best results respectively.)
Retriever MethodRetrieval(%) BLIP-2 VQA(%) LLava VQA(%)
R@5(↓) ASR*(↑) P@5(↓) ASR*(↑) EM ASR*(↑) EM ASR*(↑)
Siglip-so400mClean 44.47 - 20.12 - 21.63 - 31.96 -
X-Transfer [14] 24.42 45.09 11.01 45.28 11.79 45.49 19.93 37.64
LMM Attack [7] 5.35 87.97 1.68 91.65 11.91 44.94 11.79 63.11
Ours 4.37 90.17 1.56 92.25 5.47 74.71 12.52 60.83
CLIP ViT-HClean 43.26 - 18.78 - 20.41 - 29.77 -
AnyAttack [39] 32.68 24.46 14.09 24.97 16.16 20.82 22.96 22.88
X-Transfer [14] 13.49 68.82 4.45 76.30 5.10 75.01 11.91 59.99
LMM Attack [7] 4.86 88.77 1.43 92.39 13.00 36.31 9.48 68.16
Ours 2.31 94.66 0.51 97.28 3.28 83.93 10.45 64.90
•Recall@Kevaluates how likely the retrieved K passages
are to contain answers to the query, which is the propor-
tion of queries that have positive passages in the retrieved
results:
Recall@K= min(KX
k=1S(q, p k),1).(5)
•Precision@Kevaluates how much percent of the re-
trieved K passages contain answers to the query:
Precision@K=1
KKX
k=1S(q, p k).(6)
For generation, we adopt the corresponding metric for
each dataset. For OK-VQA, VQA score and Exact Match
are used following RA VQA [20]. For InfoSeek, ExactMatch is calculated. For each metric, we calculate the
non-target attack success rate proposed by X-transfer [14],
which is(s clean−sadv)/sclean for each metrics.
4.2. Implementation Details
We implement the proposed method based on the open-
source PyTorch [28] framework. All the experiments are
conducted on NVIDIA RTX3090. We adopt PGD-step 50,
step sizeα= 1/255, perturbation boundϵ= 8/255for the
two stages respectively in retrieval perturbation generation.
The trade-off parameterβand margin parameterγare set
to 0.4 and 0.6 in all loss functions. We adopt the X-transfer
noise asδ gin our experiments.
For OK-VQA, we use the off-the-shell CLIP ViT-L/14 as
well as a fine-tuned version of it as retrievers. For Infoseek,
we use the off-the-shell Siglip-so400m and CLIP ViT-H as

retrievers following PoisonedEye [37]. For generators, we
use the off-the-shell BLIP-2-flan-T5-xl and LLava-NEXT
models.
4.3. Baseline Models
We compare our method with various state-of-the-art visual
noise generation algorithms. AnyAttack [39] finetuned a
decoder to generate noise for any image that transforms it
to any target. We use the released AnyAttack decoder to
generate noises using the reference image in our method for
comparison. X-transfer [14] learned a transferrable noise
that can be applied to any image. We employ the ”xtrans-
ferlarge linfeps12 nontargeted” noise and setϵ= 8to
apply to all the images. LMM Attack [7] learned the noise
using PGD algorithm, with the cross entropy loss to mini-
mize similarity between the adversarial image and its image
caption as the objective function. Since the code was not re-
leased, we re-implemented the algorithm and set PGD-step
to 50.
4.4. Experimental Results
4.4.1. Main Attack Results
The main attack results of attacking the MRAG system on
two datasets are shown in Table 1 and Table 2. The overall
results show that our proposed method achieved the most
severe attack influence on retrieval and generation process
of MRAG. Several key conclusions can be made:
Our hierarchical attack works on both retrieval and
generation.Among the datasets, our hierarchical attack
method all achieve declines in both retrieval and generation
metrics. As shown in the results, former baseline methods
demonstrate attack advantage either in retrieval or gener-
ation. X-transfer [14]’s noise is more effective at attack-
ing the generator (the retrieval metrics’ decline are subtle
compared to the VQA metrics). While LMM attack [7]’s
noise is comparably more effective at attacking the retrieval
process. For our attack, the hierarchical structure leads to
a more balanced attack effectiveness on both retrieval and
generation, and disrupts the whole MRAG chain by only
modifying the image input, making the attack highly im-
perceptible to human vision while effective.
Our hierarchical attack works on the fine-tuned re-
triever.As shown in the tables, the off-the-shell models are
highly vulnerable against adversarial noise in the retrieval
task. However, previous attack methods can not achieve
equivalent effectiveness towards both off-the-shell and fine-
tuned models. For example, as shown in Table 1, the non-
hierarchical LMM attack can achieve 74.77% attack suc-
cess rate on the R@5 metric of off-the-shell CLIP, while
only getting 23.73% success rate on the fine-tuned version.
While for our hierarchical attack method, we achieved high
success rate (72.32% and 57.38%) on non-fine-tuned and
fine-tuned retrievers. Note that though the attack perfor-
Figure 4. Performance of ablated models with different steps.
mance of LMM Attack and our method are compatible on
the original CLIP, our method achieves over two times the
success rate on the fine-tuned version.
Our hierarchical attack works on various black-box
generators.We use BLIP-2 and LLaV A as black-box gen-
erators. VQA results of OK-VQA using LLaV A is shown
in Table 3. With using different black-box generators, our
attack with adversarial augmented knowledge and query
noise can all achieve damaging attack effect.
VQA
Score(↓)ASR*(↑) EM(↓) ASR*(↑)
Clean 63.30 - 67.34 -
AnyAttack 61.31 3.14 65.46 2.79
X-transfer 57.17 9.68 60.98 9.44
LMM Attack 56.72 10.39 60.42 10.23
Ours 54.19 14.39 57.77 14.21
Table 3. VQA results on OK-VQA with LLaV A as generator. Top-
5 retrieved documents are used for generation.
4.4.2. Ablation Studies
Analysis of Hierarchical Retrieval Attack.Table 4 and
Table 5 show the detailed retrieval results on OK-VQA and
Infoseek datasets. Both stages of the attack demonstrate in-
dividual effectiveness. Using stage 1 and stage 2 alone both
degrades retrieval metrics. Stage 2’s individual attack per-
formance is comparably better than stage 1, since its attack
goal is more directly targeted at retrieval. While across all
datasets and retriever evaluated, the hierarchical two-stage
attack consistently achieves optimal performance, demon-
strating the attack effect of breaking both modality and se-
mantic alignment.
Effect of PGD Steps.As shown in Figure 4 (left), we
report the Recall@5 metric on OK-VQA dataset with fine-
tuned CLIP using 10 to 50 steps of stage 1 and 2. As the
PGD algorithm increases optimization steps, the attack re-
sults become better, eventually coming to convergence. We
finally choose PGD-step50 for each stage of optimization.
Effect of Perturbation Budget.We explore the effect of
settingϵto different values. As shown in Figure 4 (right),

Table 4. Retrieval Performance on OK-VQA. (Boldand underline
represent the best and second best results respectively.)
CLIP ViT-L/14
R@5 R@10 P@5 P@10
Clean 50.57 62.25 27.86 27.48
AnyAttack [39] 50.12 61.06 27.21 26.86
X-Transfer [14] 48.14 59.02 25.27 24.73
LMM Attack [7] 12.76 19.26 4.48 4.65
Oursw/Stage 1 21.94 31.85 8.52 8.74
Oursw/Stage 2 19.70 28.22 7.56 7.77
Ours 14.00 20.77 4.86 4.93
CLIP ViT-L/14finetuned
R@5 R@10 P@5 P@10
Clean 74.63 83.31 44.03 42.27
AnyAttack [39] 74.18 82.82 43.56 41.83
X-Transfer [14] 72.69 81.25 40.99 38.91
LMM Attack [7] 56.92 68.59 27.34 26.56
Oursw/Stage 1 54.48 65.93 25.73 25.07
Oursw/Stage 2 37.65 49.68 15.56 15.78
Ours 31.81 44.19 12.40 12.78
Table 5. Retrieval Performance on Infoseek. (Boldand underline
represent the best and second best results respectively.)
Siglip-so400m
R@5 R@10 P@5 P@10
Clean 43.47 51.89 19.61 17.06
X-Transfer [14] 24.42 30.01 11.01 9.68
LMM Attack [7] 5.35 8.14 1.68 1.7
Oursw/Stage 1 5.1 8.51 1.97 1.91
Oursw/Stage 2 4.74 7.9 1.511.51
Ours 4.37 7.291.56 1.51
CLIP ViT-H
R@5 R@10 P@5 P@10
Clean 43.26 48.97 18.78 15.55
AnyAttack [39] 32.69 40.34 14.09 11.85
X-Transfer [14] 13.49 18.59 4.45 3.94
LMM Attack [7] 4.86 7.65 1.43 1.49
Oursw/Stage 1 3.77 5.95 0.92 0.89
Oursw/Stage 2 2.61 4.39 0.69 0.69
Ours 2.31 3.89 0.51 0.52
whenϵgets larger, the attack effect is more obvious. Dif-
ferent colors refer to various retrieval metrics, whenϵin-
creases, all the metrics degrade. With a limited budget
ϵ= 8, attack effectiveness can already be achieved while
Original Image
Generator 
Adversarial Image
Clean docs: 
1. macys on state street: the marshall fields department store chicago illinois united
states of america macys on state ...
2. ... the busy streets surrounding the shop , known as the jordaan, are bustling with
locals and tourists alike. ...
3. ... admittedly , it's a bit touristy and pricey , but you can't beat the location and
browsing the unique mom-and-pop shops .
4. ... state street, macy's department stor e on right, formerly marshall field's. 
chicago famous marshall fields clock ...
5. ... broadway is synonymous with theatre, fifth avenue is automatically paired with
shopping , madison avenue means the advertising industry , ...
Adversarial docs:
1. winter  sports: sledding, skiing, snowboarding, skating (for kids) - nemours
kidshealth [skip to content] open search for parents parents site sitio para padres ...
2. ... winter  sports  are lots of fun — just ask any kid who's just scored the winning
goal during an ice-hockey game or finished sledding ...
3. aug 7, 2019 · whether it is at a christmas market or end of year sale at zara, you
should expect to find people scavenging for all sorts of things. ...
4. but you need the right equipment.  flexible flyer baby pull sledbecause babies love
snow toocheck price this is a great baby sled for pulling you ...
5. baby sled – this flexible flyer baby pull sled is the perfect way to tow babies on
packed snow , groomed paths, or snow up to 4 inches deep secure support ...
LLaV A clean answer:  'Someone might go to this place to shop , as it is a busy
shopping district with a lar ge department store, Macy's, and other stores. '
LLaV A adv answer:  'Someone might go to this place to enjoy the winter  sports
activities of fered, such as sledding, skiing, snowboarding, and skating. 'Question:  Why might someone go to this place?
GT answers:  business, nyc, shop
Retrieval 
Adversarial Image
Figure 5. An example showing the original image, the adversarial
images as inputs to retriever and generator, as well as the clean
and adversarial augmented knowledge, along with the generated
answer based on them. The example demonstrates the attack effect
of our hierarchical method within imperceptible disruption.
being imperceptible.
4.4.3. Case studies
We provide a case study in Figure 5. This example shows
the attack effect of our hierarchical method under the per-
turbation budgetϵ= 8. As shown in the example, with
the original query image without noise, the retriever returns
augmented knowledge that are closely related to answering
the question. The generator provides a correct answer ac-
cordingly. However, with added imperceptible noise, the
retriever returns irrelevant augmented knowledge that are
about winter sports. With the disruptions, the final predic-
tion of the generator is wrong and unreasonable.
5. Conclusion
In this paper, we proposed a novel attack method against
MRAG pipeline that focused solely on image inputs. With
hierarchical optimization, we target MRAG’s retriever and
generator across different levels of abstraction, achieving

severe while stealth attack impact. Our research reveals
that MRAG technologies, while widely adopted in practice,
remain vulnerable to security threats posed by impercepti-
ble adversarial visual noise. Future work will focus on un-
covering deeper and more diverse potential threats brought
by visual attacks on MRAG systems and developing ro-
bust defense mechanisms to balance effectiveness and secu-
rity.
References
[1] Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi
Dehghani, Mohammadali Mohammadkhani, Bardia Mo-
hammadi, Omid Ghahroodi, Mahdieh Soleymani Baghshah,
and Ehsaneddin Asgari. Ask in any modality: A comprehen-
sive survey on multimodal retrieval-augmented generation.
arXiv preprint arXiv:2502.08826, 2025. 1, 2
[2] Jay Barach. Cross-domain adversarial attacks and robust de-
fense mechanisms for multimodal neural networks. InIn-
ternational Conference on Advanced Network Technologies
and Intelligent Computing, pages 345–362. Springer, 2024.
2
[3] Wieland Brendel, Jonas Rauber, and Matthias Bethge.
Decision-based adversarial attacks: Reliable attacks against
black-box machine learning models.arXiv preprint
arXiv:1712.04248, 2017. 3
[4] Nicholas Carlini and David Wagner. Towards evaluating the
robustness of neural networks. In2017 ieee symposium on
security and privacy (sp), pages 39–57. Ieee, 2017. 3
[5] Huanran Chen, Yichi Zhang, Yinpeng Dong, Xiao Yang,
Hang Su, and Jun Zhu. Rethinking model ensem-
ble in transfer-based adversarial attacks.arXiv preprint
arXiv:2303.09105, 2023. 3
[6] Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, So-
ravit Changpinyo, Alan Ritter, and Ming-Wei Chang.
Can pre-trained vision and language models answer vi-
sual information-seeking questions?arXiv preprint
arXiv:2302.11713, 2023. 5
[7] Xuanming Cui, Alejandro Aparcedo, Young Kyun Jang, and
Ser-Nam Lim. On the robustness of large multimodal mod-
els against image adversarial attacks. InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 24625–24634, 2024. 2, 6, 7, 8
[8] Claudio Gentile and Manfred KK Warmuth. Linear hinge
loss and average margin.Advances in neural information
processing systems, 11, 1998. 4
[9] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy.
Explaining and harnessing adversarial examples.arXiv
preprint arXiv:1412.6572, 2014. 3
[10] Zhongliang Guo, Weiye Li, Yifei Qian, Ognjen Arand-
jelovic, and Lei Fang. A white-box false positive adversar-
ial attack method on contrastive loss based offline handwrit-
ten signature verification models. InInternational Confer-
ence on Artificial Intelligence and Statistics, pages 901–909.
PMLR, 2024. 3
[11] Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios
Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-WeiChang, Daniel Kang, and Heng Ji. Mm-poisonrag: Disrupt-
ing multimodal rag with local and global poisoning attacks.
arXiv preprint arXiv:2502.17832, 2025. 1, 2
[12] Hexiang Hu, Yi Luan, Yang Chen, Urvashi Khandelwal,
Mandar Joshi, Kenton Lee, Kristina Toutanova, and Ming-
Wei Chang. Open-domain visual entity recognition: Towards
recognizing millions of wikipedia entities. InProceedings
of the IEEE/CVF International Conference on Computer Vi-
sion, pages 12065–12075, 2023. 5
[13] Wenbo Hu, Jia-Chen Gu, Zi-Yi Dou, Mohsen Fayyaz, Pan
Lu, Kai-Wei Chang, and Nanyun Peng. Mrag-bench: Vision-
centric evaluation for retrieval-augmented multimodal mod-
els.arXiv preprint arXiv:2410.08182, 2024. 1
[14] Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, and
James Bailey. X-transfer attacks: Towards super transferable
adversarial attacks on clip. InICML, 2025. 2, 6, 7, 8
[15] Andrew Ilyas, Logan Engstrom, Anish Athalye, and Jessy
Lin. Black-box adversarial attacks with limited queries and
information. InInternational conference on machine learn-
ing, pages 2137–2146. PMLR, 2018. 3
[16] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-
tau Yih. Dense passage retrieval for open-domain question
answering. InEMNLP (1), pages 6769–6781, 2020. 3
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems,
33:9459–9474, 2020. 2
[18] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.
Blip-2: Bootstrapping language-image pre-training with
frozen image encoders and large language models. InIn-
ternational conference on machine learning, pages 19730–
19742. PMLR, 2023. 1, 2
[19] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll ´ar, and C Lawrence
Zitnick. Microsoft coco: Common objects in context. In
European conference on computer vision, pages 740–755.
Springer, 2014. 5
[20] Weizhe Lin and Bill Byrne. Retrieval augmented visual
question answering with outside knowledge.arXiv preprint
arXiv:2210.03809, 2022. 5, 6
[21] Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca,
and Bill Byrne. Fine-grained late-interaction multi-modal
retrieval for retrieval augmented visual question answering.
Advances in Neural Information Processing Systems, 36:
22820–22840, 2023. 5
[22] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning.Advances in neural information
processing systems, 36:34892–34916, 2023. 1, 2
[23] Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie
Qian, Zhilin Yang, and Jie Tang. Gpt understands, too.AI
Open, 5:208–215, 2024. 1
[24] Yinuo Liu, Zenghui Yuan, Guiyao Tie, Jiawen Shi, Pan
Zhou, Lichao Sun, and Neil Zhenqiang Gong. Poisoned-
mrag: Knowledge poisoning attacks to multimodal retrieval

augmented generation.arXiv preprint arXiv:2503.06254,
2025. 1, 2
[25] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
Dimitris Tsipras, and Adrian Vladu. Towards deep learn-
ing models resistant to adversarial attacks.arXiv preprint
arXiv:1706.06083, 2017. 3
[26] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and
Roozbeh Mottaghi. Ok-vqa: A visual question answering
benchmark requiring external knowledge. InProceedings
of the IEEE/cvf conference on computer vision and pattern
recognition, pages 3195–3204, 2019. 5
[27] Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen. A
survey of multimodal retrieval-augmented generation.arXiv
preprint arXiv:2504.08748, 2025. 1, 2
[28] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer,
James Bradbury, Gregory Chanan, Trevor Killeen, Zeming
Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An im-
perative style, high-performance deep learning library.Ad-
vances in neural information processing systems, 32, 2019.
6
[29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. InInternational conference on machine learning, pages
8748–8763. PmLR, 2021. 4
[30] Christian Schlarmann and Matthias Hein. On the adversarial
robustness of multi-modal foundation models. InProceed-
ings of the IEEE/CVF International Conference on Com-
puter Vision, pages 3677–3685, 2023. 1
[31] Haodi Wang, Kai Dong, Zhilei Zhu, Haotong Qin, Aishan
Liu, Xiaolin Fang, Jiakai Wang, and Xianglong Liu. Trans-
ferable multimodal attack on vision-language pre-training
models. In2024 IEEE Symposium on Security and Privacy
(SP), pages 1722–1740. IEEE, 2024. 1
[32] Ruofan Wang, Xingjun Ma, Hanxu Zhou, Chuanjun Ji,
Guangnan Ye, and Yu-Gang Jiang. White-box multimodal
jailbreaks against large vision-language models. InProceed-
ings of the 32nd ACM International Conference on Multime-
dia, pages 6920–6928, 2024. 3
[33] Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia Shi,
Sheng Wang, Linjun Zhang, James Zou, and Huaxiu Yao.
Mmed-rag: Versatile multimodal rag system for medical vi-
sion language models.arXiv preprint arXiv:2410.13085,
2024. 1
[34] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv
preprint arXiv:2505.09388, 2025. 1
[35] Ziyi Yin, Muchao Ye, Tianrong Zhang, Jiaqi Wang, Han
Liu, Jinghui Chen, Ting Wang, and Fenglong Ma. Vqattack:
Transferable adversarial attacks on visual question answer-
ing via pre-trained models. InProceedings of the AAAI Con-
ference on Artificial Intelligence, pages 6755–6763, 2024. 1
[36] Yixiao Zeng, Tianyu Cao, Danqing Wang, Xinran Zhao,
Zimeng Qiu, Morteza Ziyadi, Tongshuang Wu, and Lei
Li. Rare: Retrieval-aware robustness evaluation forretrieval-augmented generation systems.arXiv preprint
arXiv:2506.00789, 2025. 2
[37] Chenyang Zhang, Xiaoyu Zhang, Jian Lou, Kai Wu, Zilong
Wang, and Xiaofeng Chen. Poisonedeye: Knowledge poi-
soning attack on retrieval-augmented generation based large
vision-language models. InForty-second International Con-
ference on Machine Learning. 2, 5, 7
[38] Chiyu Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, and Zhe
Liu. Adversarial attacks of vision tasks in the past 10 years:
A survey.ACM Computing Surveys, 58(2):1–42, 2025. 2
[39] Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan
Yang, Yunhao Chen, Jitao Sang, and Dit-Yan Yeung. Anyat-
tack: Towards large-scale self-supervised adversarial attacks
on vision-language models. InProceedings of the Computer
Vision and Pattern Recognition Conference, pages 19900–
19909, 2025. 2, 6, 7, 8
[40] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan
Jia.{PoisonedRAG}: Knowledge corruption attacks to
{Retrieval-Augmented}generation of large language mod-
els. In34th USENIX Security Symposium (USENIX Security
25), pages 3827–3844, 2025. 1