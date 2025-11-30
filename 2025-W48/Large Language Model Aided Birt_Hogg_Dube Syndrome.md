# Large Language Model Aided Birt-Hogg-Dube Syndrome Diagnosis with Multimodal Retrieval-Augmented Generation

**Authors**: Haoqing Li, Jun Shi, Xianmeng Chen, Qiwei Jia, Rui Wang, Wei Wei, Hong An, Xiaowen Hu

**Published**: 2025-11-25 01:55:23

**PDF URL**: [https://arxiv.org/pdf/2511.19834v1](https://arxiv.org/pdf/2511.19834v1)

## Abstract
Deep learning methods face dual challenges of limited clinical samples and low inter-class differentiation among Diffuse Cystic Lung Diseases (DCLDs) in advancing Birt-Hogg-Dube syndrome (BHD) diagnosis via Computed Tomography (CT) imaging. While Multimodal Large Language Models (MLLMs) demonstrate diagnostic potential fo such rare diseases, the absence of domain-specific knowledge and referable radiological features intensify hallucination risks. To address this problem, we propose BHD-RAG, a multimodal retrieval-augmented generation framework that integrates DCLD-specific expertise and clinical precedents with MLLMs to improve BHD diagnostic accuracy. BHDRAG employs: (1) a specialized agent generating imaging manifestation descriptions of CT images to construct a multimodal corpus of DCLDs cases. (2) a cosine similarity-based retriever pinpointing relevant imagedescription pairs for query images, and (3) an MLLM synthesizing retrieved evidence with imaging data for diagnosis. BHD-RAG is validated on the dataset involving four types of DCLDs, achieving superior accuracy and generating evidence-based descriptions closely aligned with expert insights.

## Full Text


<!-- PDF content starts -->

Large Language Model Aided Birt-Hogg-Dubé
Syndrome Diagnosis with Multimodal
Retrieval-Augmented Generation
Haoqing Li, Jun Shi, Xianmeng Chen, Qiwei Jia, Rui Wang, Wei Wei, Hong
An, Xiaowen Hu
School of Computer Science and Technology
li_haoqing@mail.ustc.edu.cn,
Department of Pulmonary and Critical Care Medicine; Center for Diagnosis and
Management of Rare Diseases, the First Affiliated Hospital of USTC, Division of Life
Sciences and Medicine, USTC,
WanNan Medical College
Abstract.Deep learning methods face dual challenges of limited clini-
cal samples and low inter-class differentiation among Diffuse Cystic Lung
Diseases (DCLDs) in advancing Birt-Hogg-Dubé syndrome (BHD) di-
agnosis via Computed Tomography (CT) imaging. While Multimodal
Large Language Models (MLLMs) demonstrate diagnostic potential for
such rare diseases, the absence of domain-specific knowledge and refer-
able radiological features intensify hallucination risks. To address this
problem, we propose BHD-RAG, a multimodal retrieval-augmented gen-
eration framework that integrates DCLD-specific expertise and clinical
precedents with MLLMs to improve BHD diagnostic accuracy. BHD-
RAG employs: (1) a specialized agent generating imaging manifestation
descriptions of CT images to construct a multimodal corpus of DCLDs
cases. (2) a cosine similarity-based retriever pinpointing relevant image-
description pairs for query images, and (3) an MLLM synthesizing re-
trieved evidence with imaging data for diagnosis. BHD-RAG is validated
on the dataset involving four types of DCLDs, achieving superior ac-
curacy and generating evidence-based descriptions closely aligned with
expert insights.
Keywords:Birt-Hogg-DubésyndromeDiagnosis·Retrieval-Augmented
Generation·Multimodal Large Model·Computed Tomography.
1 Introduction
Birt-Hogg-Dubé syndrome (BHD) is an autosomal dominant disorder charac-
terised clinically by skin fibrofolliculomas, pulmonary cysts, spontaneous pneu-
mothorax, and renal cancer [1,2]. The BHD diagnosis relies on identifying cystic
lesionsandpneumothoraxinchestComputedTomography(CT)scans.However,
it remains challenging to distinguish BHD from other Diffuse Cystic Lung Dis-
eases (DCLDs) with similar radiological features, such as numerous diffuselyarXiv:2511.19834v1  [cs.CV]  25 Nov 2025

2 Anonymized Author et al.
(a) (b) (c) (d)
Fig. 1.Representative CT imaging examples of four challenging-to-differentiate
DCLDs: (a) BHD, characterized by cysts of variable size, elliptical or flattened, pre-
dominantly located in the lower lungs and subpleural regions; (b) LAM, featuring
round and of similar size cysts; (c) PLCH, predominantly located in the lower lungs
and subpleural regions; (d) LIP, the cysts vary in size, predominantly in the bilateral
lung bases, and follow a perivascular pattern.
distributed, thin-walled, round, or irregular pulmonary cystic lesions. These
DCLDs mainly include Lymphangioleiomyomatosis (LAM), Pulmonary Langer-
hans Cell Histiocytosis (PLCH), and Lymphocytic Interstitial Pneumonia (LIP)
[2,3]. Therefore, accurate differentiation between BHD and other DCLDs is crit-
ical for precise diagnosis and appropriate clinical management.
In recent years, deep learning methods have demonstrated remarkable per-
formance in challenging diagnostic tasks, comparable to human experts [4–6].
However, existing task-specific methods often rely on large-scale training data
and do not apply to rare diseases such as BHD with limited clinical samples.
Furthermore, the low inter-class differentiation in imaging manifestations among
DCLDs hinder traditional discriminative methods from optimizing complex de-
cision boundaries, limiting their classification performance for BHD.
With the In-Context Learning (ICL) paradigm [7–10], Large Language Mod-
els (LLMs) have shown promise in data-scarce medical scenarios. For example,
GPT-4 outperform domain-specific models like Med-PaLM 2 [11] without requir-
ing task-specific training. [12]. This technological advancement positions Mul-
timodal Large Language Models (MLLMs) as a feasible solution for achieving
accurate diagnosis of BHD syndrome in low-sample clinical settings. However,
the susceptibility of MLLMs to "hallucination", i.e. generating plausible yet
incorrect outputs, is a critical limitation [13]. The absence of domain-specific
expertise and referable radiological features of DCLDs further intensify halluci-
nation risks, as shown in Fig. 5.
Retrieval-Augmented Generation (RAG) integrates information retrieval and
text generation, leveraging external corpus to enhance MLLMs in knowledge-
intensive tasks [14]. As a typically knowledge-intensive field, medical diagnostics
can benefit from RAG, enabling MLLMs to generate evidence-based medical
responses and analyze complex queries [15,16]. For BHD diagnosis, RAG is ca-
pable of providing domain-specific knowledge and imaging features on BHD and
other DCLDs, addressing the hallucination inherent in MLLMs. However, the
rarity of DCLDs and the limited availability of publicly accessible information
constrain the construction of corpus.

BHD Diagnosis with Multimodal Retrieval-Augmented Generation 3
Query Image
BHD
LAM
LIP
PLCH
DCLD -specific ExpertisePlease refer to the clinical precedents, 
describe the imaging features, and 
determine whether it is BHD.User 
User 
User 
User 
Multiple thin -walled cystic 
lesions in both  lungs  … 
Therefore, a diagnosis of BHD 
may be considered.Multiple thin -walled cystic 
lesions in both  lungs  … 
Therefore, a diagnosis of BHD 
may be considered.
,,t s cI I I
      
Clinical DCLDs Acquisition
Transverse
Transverse
PLCH
LAMBHDLIP
LAMBHDLIP
Respiratory 
Medicine
sISagittalSagittal
sISagittal
Coronal
Coronal
Clinical DCLDs Acquisition
Transverse
PLCH
LAMBHDLIP
Respiratory 
Medicine
sISagittal
Coronal
Multiple cavitary lesions can be seen in both 
lungs. These lesions vary in size and round in 
shape . …… Based on these imaging features, 
the diagnosis of B HD (Birt Hogg Dubé  syndrome) 
may be considered.Multiple cavitary lesions can be seen in both 
lungs. These lesions vary in size and round in 
shape . …… Based on these imaging features, 
the diagnosis of B HD (Birt Hogg Dubé  syndrome) 
may be considered.MLLM
MLLM
Initial Descriptions
' ' ',,
t s cT T TInitial Descriptions
' ' ',,
t s cT T T
Experts Revise
 Experts Revise
Multiple cystic lesions or cysts can be seen in 
both lungs. These lesions vary in size and shape, 
including round, oval and flat.  …… Based on 
these imaging features, the diagnosis of B HD 
(Birt Hogg Dubé  syndrome) may be considered.Multiple cystic lesions or cysts can be seen in 
both lungs. These lesions vary in size and shape, 
including round, oval and flat.  …… Based on 
these imaging features, the diagnosis of B HD 
(Birt Hogg Dubé  syndrome) may be considered.
Revised Descriptions
,,t s cT T TRevised Descriptions
,,t s cT T T
Retriever
Retriever
Top-𝑘 Retrived  Pairs
 Top-𝑘 Retrived  Pairs
(a) DCLD -Corpus Construction (b) BHD -RAG
MLLM
Image -Description Pairs
DCLD -Corpus
Image -Description Pairs
DCLD -Corpus
Fig. 2.An overview of the proposed BHD-RAG framework.
To address these challenges, we propose BHD-RAG, a multimodal retrieval-
augmented generation framework that integrates DCLD-specific expertise and
clinical precedents with MLLMs to enhance BHD diagnosis. As shown in Fig. 2,
in response to the scarcity of external knowledge to construct the RAG system,
we sourced clinical cases of DCLDs from the respiratory department to estab-
lish a dedicated DCLD-Corpus. MLLMs are employed to generate respiratory
medicine-specific descriptions from CT slices, which are refined in collaboration
with respiratory specialists to construct a corpus of image-description pairs.
Moreover, to overcome the challenge of optimizing the decision boundary be-
tween BHD and other DCLDs, BHD-RAG employs a cosine-space similarity re-
trievertoexpandtheangularmarginandextractpertinentpairsfromthecorpus.
The generator subsequently integrates the knowledge retrieved with the query to
produce diagnostic responses. BHD-RAG is validated on the dataset encompass-
ingfourDCLDs,demonstratingsuperioraccuracyandgeneratingevidence-based
descriptions consistent with expert assessments.
2 Methodology
2.1 DCLD-Corpus Construction
MLLMs are prone to struggling with factual accuracy when addressing challeng-
ing medical tasks. Retrieval-Augmented Generation (RAG) offers a solution by
integrating external corpus to guide response generation and improve reliabil-
ity. As a foundational step, it is essential to develop a DCLD-corpus for BHD
diagnosis, to support the RAG system.
To develop a corpus for DCLDs, it is essential to gather extensive image-
description paired diagnostic examples. We leverage the fine-grained image de-

4 Anonymized Author et al.
scriptioncapabilitiesofMLLMstogeneratedetailedrespiratorymedicine-specific
descriptions. Firstly, We segment the CT scans into 2D slices along the coronal,
sagittal, and transverse planes{I t, Is, Ic}, and retain abnormal slices containing
prominent cystic lesions. Moreover, GPT-4-turbo [17] is used to generate initial
imaging manifestation descriptions{T′
t, T′
s, T′
c}of these slices. Medical experts
then refine these descriptions, correcting misclassifications to ensure the accu-
racy of the descriptions{T t, Ts, Tc}and the alignment with expert diagnostic
logic, as shown in Fig. 2 (a).
The descriptions{T t, Ts, Tc}generated by the experts-assisted MLLMs are
paired with the corresponding images to construct the corpus:
{Ii
corpus ,Ti
corpus}={(Ii
t, Ti
t),(Ii
s, Ti
s),(Ii
c, Ti
c)},(1)
wherei= 0,1,2, ..., n−1, andnrepresents the number of pairs. Due to the
limited availability of clinical samples for rare DCLDs, related data is scarce in
training databases of MLLMs. Thus, maximizing the utility of our valuable clin-
ical samples and provides the MLLM with comprehensive external knowledge
of DCLDs is imperative. Moreover, the BHD diagnosis requires both an assess-
ment of lesion morphology and the determination of its spatial location within
the lungs. Multi-view imaging provides a comprehensive features of lesion mor-
phology than single-view imaging, offering spatial evidence from multiple per-
spectives for MLLMs.
Furthermore, the lack of DCLD-specific knowledge in MLLMs introduces
misleading contexts and extraneous information, compromising generative per-
formance [18–20]. To address this, DCLD-corpus is designed to emphasize dis-
tinctive DCLDs features, providing fine-grained comparative analyses to reduce
domain knowledge ambiguity, as shown in Fig. 2 (b). The constructed corpus
integrates multimodal, multi-grained DCLD-specific knowledge, enabling the re-
triever to supply the MLLMs with the most relevant and precise information.
2.2 Cosine-Space Similarity Retriever
The retriever is integral to the BHD-RAG framework, tasked with extracting
the minimal corpus subset essential for accurate inference by the generator [21].
During the multimodal corpus retrieval, the BHD-RAG retriever selects the top-
kimage-description pairs, which are most similar to the query CT slide. These
references contain expert-curated domain-specific knowledge on DCLDs rare dis-
eases and guide the MLLM in generating accurate responses to DCLDs-related
queries. Ensuring retrieval accuracy and efficiency is crucial to this process.
The retriever is trained with a similarity-based metric learning paradigm
[22,23], as illustrated in Fig. 3. Both positive and negative samples are matched
within the same anatomical view. Positive pairs, e.g.(Ii
BHD, Ij
BHD ), are con-
structed by randomly sampling different CT slices from the same class, while
negative pairs, e.g.(Ii
BHD, Ij
nonBHD ), are generated from different classes.Ii
BHD
represents a randomly selectedi-th sample from the BHD slides. The outputy
of the retrieverRcan be formulated as follows:

BHD Diagnosis with Multimodal Retrieval-Augmented Generation 5
Retriver
EncoderBHD BHD
BHD Non-BHDPositive Pairs
Negative PairsBHDNon-BHDAngular  MarginQuery image
( , )ij
BHD BHDII
( , )ij
BHD nonBHDII
( , )ij
BHD nonBHDyy
( , )ij
BHD BHDyy
cos( , )ij
BHD BHDy y m −
) cos( ,ij
BHD nonBHDyy−
Fig. 3.The proposed cosine spatial similarity measure retriever.
(yi
BHD, yj
nonBHD ) =R(Ii
BHD, Ij
nonBHD ).(2)
Under the substantial overlap of imaging features across various DCLDs,
conventional approaches relying on cosine similarity with Softmax struggle to
achievepreciseclassificationincosinespace,asshowninTable1.Toaddressthis,
we define the decision boundaries in the cosine space, and introduce CosFace loss
to enhance the angular margin between BHD and non-BHD classes [24]:
L=1
NX
i−loges(cos(yi
BHD,yj
BHD)−m)
es(cos(yi
BHD,yj
BHD)−m)+escos(yi
BHD,yj
nonBHD),(3)
whereNdenotes the total number of positive and negative pairs,cos(.)repre-
sents the cosine similarity between sample logits,sis the scaling factor, andm
is the angular margin between classes.
2.3 Retrieval-Augmented Generation for BHD Diagnosis
The objective is to diagnose BHD from the query imageI qand generate an
accurate description of its imaging characteristics, producing response ˜A. To
enhance the ability in describing and discriminating DCLDs features, the top-
kimage-description pairs{I topk,Ttopk}retrieved from the corpus are combined
with expert-curated DCLD-specific expertise{I e, Te}. This integration strength-
enstheperformanceoftheMLLMsinknowledge-intensiveBHDdiagnostictasks.
The output of BHD-RAG is formalized as:
{Itopk,Ttopk}= arg min
Itopk∈Icorpus
|Itopk|=k
cos 
R(Iq,Ii
corpus )	
, i= 0,1, . . . , n−1,(4)
˜A=f(I q, Tq;M,I corpus ,Tcorpus ) = arg min
APM(A|I q, Ie,Itopk, Tq, Te,Ttopk),
(5)
wheref(.)represents the proposed BHD-RAG,T qis the input prompt,Mde-
notes the MLLM, ˜AandAare the predicted response and the ground truth,
respectively.

6 Anonymized Author et al.
Table 1.Performance comparison of our approach with the comparing methods.
MethodsAccuracy Precision Recall F1 Specificity
ResNet-50-3D [25] 0.5789 0.5714 0.8000 0.6667 0.3333
DenceNet-121-3D [26] 0.6316 0.6154 0.8000 0.6957 0.4444
ResNext-50-3D [27] 0.6316 0.6154 0.8000 0.6957 0.4444
MedicalNet [28] 0.6316 0.7143 0.5000 0.5882 0.7778
M3T [29] 0.6842 0.7000 0.7000 0.7000 0.6667
LLaVA-Med [30] 0.4211 0.0000 0.0000 0.00000.8889
GPT-4o [17] 0.5263 0.52940.90000.6667 0.1111
BHD-RAG + GPT-4o0.6842 0.7000 0.7000 0.7000 0.6667
GPT-4-turbo [17] 0.6316 0.5882 0.7407 0.6316 0.2222
BHD-RAG + GPT-4-turbo 0.7895 0.80000.80000.80000.7778
3 Experiments and Results
3.1 Data Curation and Implementation Details
CTscansfrompatientswithconfirmeddiagnosesofDCLDsbasedonhistopatho-
logical confirmation or in accordance with well-established professional society
guidelines, were acquired from the **** Hospital. The contiguous DICOM im-
ages, with a median slice thickness of 1.25 mm (range: 1–5 mm), comprise 97
cases (50 BHD, 18 LAM, 7 PLCH, and 22 LIP).
These data are divided into train (retrieve) and test (query) set in an 8 :
2 ratio at patient-level, with slicesI t, Is, Iccontaining typical DCLDs lesions
identified. To ensure diversity in the corpus, adjacent key slices were spaced by
at least two frames. Additionally, slices exhibiting typical lesions are prioritized
to maximize diagnostic information. The numbers of typical slices for BHD,
LAM, PLCH, and LIP were 753, 255, 107, and 329, respectively. Then, image-
description pairs are acquired as outlined in Section 2. The train set is used
both for retriever training and as image-modality knowledge incorporated into
the corpus.
To ensure the response efficiency, ResNet-18 [25] was conducted as the re-
triever, which is trained for 500 epochs using the AdamW optimizer with an
initial learning rate of1.0×10−4and a cosine learning rate decay strategy. The
weight decay is set to1.0×10−4, and the batch size is 32. All the slices were
normalized using a lung window and resized to 256×256 for training, while
the corpus is composed of the 369×369 resolution. The proposed approach is
implemented in PyTorch, and all the methods are conducted on 8×NVIDIA
RTX 4090 GPUs.
3.2 Qualitative and Quantitative Results
As shown in Table 1 and Fig. 4, we evaluate the performance of BHD-RAG
on the DCLDs dataset using five metrics: accuracy, precision, recall, F1-score,

BHD Diagnosis with Multimodal Retrieval-Augmented Generation 7
(a) (b)
Fig. 4.Theproposedcosinespatialsimilaritymeasureretriever.(a)Influenceofkvalue
on BHD-RAG. (b) Quantitative comparison between BHD-RAG and other methods.
Table 2.Ablation experiments of our BHD-RAG on the DCLDs dataset. To observe
the performance changes, we remove the retriever and typical features.
RetrieverTypical
FeaturesAccuracy Precision Recall F1 Specificity
0.6316 0.5882 0.7407 0.6316 1.0000
0.6842 0.7500 0.6000 0.6667 0.6000
0.6842 0.6667 0.8000 0.7273 0.8000
0.7895 0.8000 0.8000 0.8000 0.8000
and specificity. Our approach significantly outperforms the best discriminant
method, M3T, with metric enhancements ranging from 14.29 % (precision / re-
call / F1-score) to 19.99 % (specificity), while attaining a 15.39 % accuracy im-
provement. Additionally, it demonstrates an advantage over MLLMs. BHD-RAG
significantly enhances GPT-4o, improving accuracy, precision, and F1-score by
30.00%, 32.23%, and 4.99%, respectively. When integrated with GPT-4-turbo,
these gains further increase to 25%, 36.01%, and 26.66%. In contrast, MLLMs
without retrieval-augmented generation, particularly GPT-4o, tends to classify
all DCLDs samples as BHD, while LLaVA-Med categorizes all DCLDs cases
as non-BHD. Although these misclassifications leads to superficially improved
recall or Specificity, it lacks clinical significance.
Fig. 4 (a) illustrates the impact ofkon the performance of BHD-RAG. A
largerkenhances stability andk= 12achieves the best performance. Addition-
ally, qualitative comparison of BHD-RAG and GPT-4-turbo are presented in
Fig. 5. BHD-RAG enables precise diagnosis of DCLDs and generates evidence-
based imaging descriptions, emphasizing subtle features that may be overlooked
by experts and MLLMs. In contrast, models without BHD-RAG generate in-
accurate, inadequate, hallucination-prone descriptions with colloquial language
and insufficient clinical precision.

8 Anonymized Author et al.
GPT-4-turbo
GPT-4-turbo
BHD -RAG
The provided CT images reveal multiple well -defined, thin -walled cystic lesions of varying sizes, predominantly distributed near the 
mediastinum and along the interlobar fissures in both lungs. These imaging features are characteristic of Birt -Hogg -Dubé  syndrom e 
(BHD), particularly the multiplicity and distribution pattern of the cystic lesions. Therefore, the diagnosis is consistent with BHD.
The image reveals thefollowing keyfeatures :Cystic Changes :Multiple cystic lesions ofvarying sizes arepresent inthelungs,
appearing asthin-walled, well-defined air-filled spaces .Pulmonary Parenchyma :Inaddition tocystic structures ,patchy ground -
glass opacities areobserved, which may indicate coexisting pulmonary pathology orinfection .
Based onthese imaging characteristics, apreliminary differential diagnosis includes :BHD :Characterized byevenly distributed,
thin-walled cysts, predominantly located inthecentral and basal lung regions .LAM :Features diffusely distributed small cysts,
often associated with lung parenchymal destruction .PLCH :Presents with cysts ofvarying sizes, often accompanied bynodular
lesions .LIP:Typically manifests asdiffuse reticular opacities ornodular patterns .
Considering the distribution and morphology of the cystic lesions in this image, BHD is the most likely diagnosis , as the cysts are 
uniformly distributed and exhibit thin, well -defined walls, consistent with BHD imaging characteristics.
The imaging reveals multiple cystic lesions ofvarying sizes inboth lungs, exhibiting round orirregular shapes with thin-walls
andarelatively uniform distribution .Some cysts arealigned along thebronchovascular bundles, accompanied bymultiple nodules .
Nosignificant cyst fusion orlarge bullae formation isobserved .Based onthese imaging characteristics, lymphocytic interstitial
pneumonia (LIP) isalikely diagnosis .
The CTscan reveals across -sectional view ofthelungs, demonstrating multiple cystic lesions ofvarying sizes, predominantly
distributed intheupper and middle lung fields .These cysts exhibit irregular shapes and aresurrounded bythickened
interstitial tissue .The imaging findings aremost consistent with Birt-Hogg -Dubé syndrome (BHD), which ischaracterized by
multiple, asymmetrically distributed pulmonary cysts ,often located near thepleura and mediastinum .Compared toother cystic
lung diseases such aslymphangioleiomyomatosis (LAM), pulmonary Langerhans cell histiocytosis (PLCH), and lymphocytic
interstitial pneumonia (LIP), BHD typically presents with larger, irregularly shaped cysts predominantly intheupper and
middle lung zones .Conclusion :Theimaging features—multiple cystic lesions ofvarying sizes, primarily intheupper and middle
lung fields, with irregular margins andsurrounding interstitial thickening —areindicative ofBHD .
Imaging shows diffusely distributed small nodules without cystic orbullous lesions, lacking regional predilection .Such
features donotalign with thetypical radiological manifestations ofBirt-Hogg -Dubé syndrome (BHD) .Given thepattern observed,
pulmonary Langerhans cell histiocytosis (PLCH) should beconsidered ,asitoften presents with multiple small nodules,
particularly intheupper lung fields, sometimes progressing tocavitation .However, definitive diagnosis requires correlation with
clinical data andfurther investigations .Conclusion :NotBHD .
The CT scan primarily depicts the lungs, revealing multiple variably sized, well -defined radiolucent areas, suggestive of 
pulmonary cysts . Based on the provided classification, these imaging characteristics most closely align with Birt -Hogg -Dubé  
syndrome (BHD). BHD typically presents with multiple pulmonary cysts, predominantly distributed in the mid -lung and 
basal regions, with relative sparing of the perihilar area. The described features include multiple scattered cystic structur es,
primarily located in the peripheral lung zones. Conclusion: BHD.
BHD -RAG
GPT-4-turbo
BHD -RAGBHD
LIP
PLCH
Fig. 5.Qualitative comparison of BHD-RAG and GPT-4-turbo diagnoses. Errors or
imprecisions in results are highlighted in red by experts.
3.3 Ablation Study
We conduct an ablation study on BHD-RAG to verify the effectiveness of the
proposedcosine-spacesimilarityretrieverandtypicalfeatures.AsshowninTable
2, removing the retriever or typical features reduces accuracy by 13.34 %, while
removing both results in a 20 % decline. Individually retrieved image-description
pairs exhibit limited discriminative features, rendering them suboptimal as ex-
ternal knowledge for DCLDs diagnosis. In contrast, integrating typical differ-
entiating features of DCLDs enhances alignment with clinical diagnoses. This
confirms that BHD-RAG enhances the capacity of MLLMs to discern domain-
specific DCLDs features through external knowledge integration.
4 Conclusion
This study proposes BHD-RAG, a retrieval-augmented generation framework
designed for BHD diagnosis. By leveraging a cosine-space similarity retriever,
the framework integrates DCLD-specific expertise and clinical precedents with
MLLMs to mitigate the inherent hallucination and improve BHD diagnostic
accuracy. Evaluation results on the DCLDs dataset demonstrate the effective-
ness and generalizability of the proposed method. Future works will focus on
expanding the clinical dataset through multi-center collaborations to develop a
multimodal foundation model for DCLDs.

BHD Diagnosis with Multimodal Retrieval-Augmented Generation 9
References
1. Menko, F.H., Van Steensel, M.A., Giraud, S., Friis-Hansen, L., Richard, S., Ungari,
S., Nordenskjöld, M., vO Hansen, T., Solly, J., Maher, E.R.: Birt-hogg-dubé syn-
drome: diagnosis and management. The lancet oncology10(12), 1199–1206 (2009)
2. Hu, D., Wang, R., Liu, J., Chen, X., Jiang, X., Xiao, J., Ryu, J.H., Hu, X.: Clin-
ical and genetic characteristics of 100 consecutive patients with birt-hogg-dubé
syndrome in eastern chinese region. Orphanet Journal of Rare Diseases19(1), 348
(2024)
3. Group,E.C.:Expertconsensusonthediagnosisandmanagementofbirt-hogg-dubé
syndrome. Chinese journal of tuberculosis and respiratory diseases46(9), 897–908
(2023)
4. Ma, J., He, Y., Li, F., Han, L., You, C., Wang, B.: Segment anything in medical
images. Nature Communications15(1), 654 (2024)
5. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomed-
ical image segmentation. In: Medical image computing and computer-assisted
intervention–MICCAI 2015: 18th international conference, Munich, Germany, Oc-
tober 5-9, 2015, proceedings, part III 18. pp. 234–241. Springer (2015)
6. Goh, E., Gallo, R., Hom, J., Strong, E., Weng, Y., Kerman, H., Cool, J.A., Kanjee,
Z., Parsons, A.S., Ahuja, N., et al.: Large language model influence on diagnos-
tic reasoning: a randomized clinical trial. JAMA Network Open7(10), e2440969–
e2440969 (2024)
7. Dai, D., Sun, Y., Dong, L., Hao, Y., Ma, S., Sui, Z., Wei, F.: Why can gpt learn in-
context?languagemodelssecretlyperformgradientdescentasmeta-optimizers.In:
Findings of the Association for Computational Linguistics: ACL 2023. pp. 4005–
4019 (2023)
8. Liu, Y., Liu, J., Shi, X., Cheng, Q., Huang, Y., Lu, W.: Let’s learn step by step:
Enhancing in-context learning ability with curriculum learning. arXiv preprint
arXiv:2402.10738 (2024)
9. Wang, F., Shao, P., Zhang, Y., Yu, B., Liu, S., Ding, N., Cao, Y., Kang, Y.,
Wang, H.: Omnirl: In-context reinforcement learning by large-scale meta-training
in randomized worlds. arXiv preprint arXiv:2502.02869 (2025)
10. Dong, Q., Li, L., Dai, D., Zheng, C., Ma, J., Li, R., Xia, H., Xu, J., Wu, Z., Liu,
T., et al.: A survey on in-context learning. arXiv preprint arXiv:2301.00234 (2022)
11. Singhal, K., Tu, T., Gottweis, J., Sayres, R., Wulczyn, E., Amin, M., Hou, L.,
Clark, K., Pfohl, S.R., Cole-Lewis, H., et al.: Toward expert-level medical question
answering with large language models. Nature Medicine pp. 1–8 (2025)
12. Nori, H., Lee, Y.T., Zhang, S., Carignan, D., Edgar, R., Fusi, N., King, N., Larson,
J., Li, Y., Liu, W., et al.: Can generalist foundation models outcompete special-
purpose tuning? case study in medicine. arXiv preprint arXiv:2311.16452 (2023)
13. Ji,Z.,Lee,N.,Frieske,R.,Yu,T.,Su,D.,Xu,Y.,Ishii,E.,Bang,Y.J.,Madotto,A.,
Fung, P.: Survey of hallucination in natural language generation. ACM Computing
Surveys55(12), 1–38 (2023)
14. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang,
H.: Retrieval-augmented generation for large language models: A survey. arXiv
preprint arXiv:2312.10997 (2023)
15. Wu, J., Zhu, J., Qi, Y., Chen, J., Xu, M., Menolascina, F., Grau, V.: Medical graph
rag: Towards safe medical large language model via graph retrieval-augmented
generation. arXiv preprint arXiv:2408.04187 (2024)

10 Anonymized Author et al.
16. Xiong, G., Jin, Q., Wang, X., Zhang, M., Lu, Z., Zhang, A.: Improving retrieval-
augmented generation in medicine with iterative follow-up questions. In: Biocom-
puting 2025: Proceedings of the Pacific Symposium. pp. 199–214. World Scientific
(2024)
17. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida,
D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv
preprint arXiv:2303.08774 (2023)
18. Mao, K., Liu, Z., Qian, H., Mo, F., Deng, C., Dou, Z.: Rag-studio: Towards in-
domain adaptation of retrieval augmented generation through self-alignment. In:
Findings of the Association for Computational Linguistics: EMNLP 2024. pp. 725–
735 (2024)
19. Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang,
P.: Lost in the middle: How language models use long contexts. Transactions of
the Association for Computational Linguistics12, 157–173 (2024)
20. Yoran, O., Wolfson, T., Ram, O., Berant, J.: Making retrieval-augmented language
models robust to irrelevant context. In: The Twelfth International Conference on
Learning Representations (2024)
21. Cuconasu, F., Trappolini, G., Siciliano, F., Filice, S., Campagnano, C., Maarek,
Y., Tonellotto, N., Silvestri, F.: The power of noise: Redefining retrieval for rag
systems. In: Proceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval. pp. 719–729 (2024)
22. Kaya, M., Bilge, H.Ş.: Deep metric learning: A survey. Symmetry11(9), 1066
(2019)
23. Li, X., Yang, X., Ma, Z., Xue, J.H.: Deep metric learning for few-shot image classi-
fication: A review of recent developments. Pattern Recognition138, 109381 (2023)
24. Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., Li, Z., Liu, W.: Cosface:
Large margin cosine loss for deep face recognition. In: Proceedings of the IEEE
conference on computer vision and pattern recognition. pp. 5265–5274 (2018)
25. He,K.,Zhang,X.,Ren,S.,Sun,J.:Deepresiduallearningforimagerecognition.In:
Proceedings of the IEEE conference on computer vision and pattern recognition.
pp. 770–778 (2016)
26. Oktay, O., Schlemper, J., Folgoc, L.L., Lee, M., Heinrich, M., Misawa, K., Mori,
K., McDonagh, S., Hammerla, N.Y., Kainz, B., et al.: Attention u-net: Learning
where to look for the pancreas. arXiv preprint arXiv:1804.03999 (2018)
27. Xie,S.,Girshick,R.,Dollar,P.,Tu,Z.,He,K.:Aggregatedresidualtransformations
for deep neural networks. In: Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition (CVPR) (July 2017)
28. Chen, S., Ma, K., Zheng, Y.: Med3d: Transfer learning for 3d medical image anal-
ysis. arXiv preprint arXiv:1904.00625 (2019)
29. Jang, J., Hwang, D.: M3t: three-dimensional medical image classifier using multi-
plane and multi-slice transformer. In: Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition. pp. 20718–20729 (2022)
30. Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T.,
Poon, H., Gao, J.: Llava-med: Training a large language-and-vision assistant for
biomedicine in one day. Advances in Neural Information Processing Systems36,
28541–28564 (2023)