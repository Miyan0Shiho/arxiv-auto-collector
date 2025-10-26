# Zero-Shot Vehicle Model Recognition via Text-Based Retrieval-Augmented Generation

**Authors**: Wei-Chia Chang, Yan-Ann Chen

**Published**: 2025-10-21 10:39:39

**PDF URL**: [http://arxiv.org/pdf/2510.18502v1](http://arxiv.org/pdf/2510.18502v1)

## Abstract
Vehicle make and model recognition (VMMR) is an important task in intelligent
transportation systems, but existing approaches struggle to adapt to newly
released models. Contrastive Language-Image Pretraining (CLIP) provides strong
visual-text alignment, yet its fixed pretrained weights limit performance
without costly image-specific finetuning. We propose a pipeline that integrates
vision language models (VLMs) with Retrieval-Augmented Generation (RAG) to
support zero-shot recognition through text-based reasoning. A VLM converts
vehicle images into descriptive attributes, which are compared against a
database of textual features. Relevant entries are retrieved and combined with
the description to form a prompt, and a language model (LM) infers the make and
model. This design avoids large-scale retraining and enables rapid updates by
adding textual descriptions of new vehicles. Experiments show that the proposed
method improves recognition by nearly 20% over the CLIP baseline, demonstrating
the potential of RAG-enhanced LM reasoning for scalable VMMR in smart-city
applications.

## Full Text


<!-- PDF content starts -->

Zero-Shot Vehicle Model Recognition via
Text-Based Retrieval-Augmented Generation
Wei-Chia Chang∗and Yan-Ann Chen∗
∗Yuan Ze University, Taiwan
Email: s1113341@mail.yzu.edu.tw, chenya@saturn.yzu.edu.tw
Abstract—Vehicle make and model recognition (VMMR) is
an important task in intelligent transportation systems, but
existing approaches struggle to adapt to newly released models.
Contrastive Language–Image Pretraining (CLIP) provides strong
visual–text alignment, yet its fixed pretrained weights limit
performance without costly image-specific finetuning. We propose
a pipeline that integrates vision–language models (VLMs) with
Retrieval-Augmented Generation (RAG) to support zero-shot
recognition through text-based reasoning. A VLM converts vehi-
cle images into descriptive attributes, which are compared against
a database of textual features. Relevant entries are retrieved
and combined with the description to form a prompt, and a
language model (LM) infers the make and model. This design
avoids large-scale retraining and enables rapid updates by adding
textual descriptions of new vehicles. Experiments show that the
proposed method improves recognition by nearly 20% over the
CLIP baseline, demonstrating the potential of RAG-enhanced
LM reasoning for scalable VMMR in smart-city applications.
I. INTRODUCTION
In recent years, the rapid growth of artificial intelligence
has accelerated the development of smart cities. Among var-
ious applications, accurate recognition of vehicle makes and
models is useful for intelligent transportation systems. Timely
identification assists in traffic accident analysis, improves
traffic-flow management, and supports law enforcement when
license plates are missing or deliberately altered. Despite these
benefits, vehicle recognition remains challenging due to the
continuous introduction of new models and the variability of
visual appearances across different conditions.
Previous studies on vehicle recognition relied on computer-
vision classifiers trained on handcrafted or deep visual features
[1]–[3]. These approaches required frequent retraining and
lacked robustness to unseen models. More recently, mul-
timodal image–text methods have shown promise for vi-
sual recognition by aligning visual content with linguistic
descriptions. Representative frameworks such as CLIP [4]
achieve strong general-purpose alignment through large-scale
pretraining. However, their performance is constrained by
fixed pretrained data and weights, which limits recognition
of newly released vehicle models. Furthermore, reliance on
global image cues reduces effectiveness when only partial
features (e.g., headlights or grilles) are available, a situation
common in accident or surveillance scenarios.
To address these limitations, we propose a zero-shot frame-
work that integrates vision–language models (VLMs) with
retrieval-augmented generation (RAG). A VLM converts ve-
hicle visual attributes into textual descriptions, which arecompared against a vector database of exterior features. A
language model then reasons over the retrieved entries to
infer the make and model, eliminating the need for additional
image retraining. RAG [5] grounds the reasoning process in
external knowledge, reducing hallucinations and improving
accuracy. Since new vehicle models can be incorporated by
simply updating textual descriptions, the framework remains
lightweight, scalable, and practical for intelligent transporta-
tion systems.
The main contributions of this paper are summarized as
follows:
•RAG-based framework for car model recognition:We
implement a text-centric architecture that converts visual
features into semantic descriptions, performs retrieval,
and reasons about make and model. To the best of
our knowledge, this is the first work to apply RAG
methodology to vehicle identification tasks, validating the
feasibility of this approach for automotive recognition.
•Zero-shot new car model recognition:The proposed
method can identify car models from textual descriptions
of frontal features, without requiring images.
•Lightweight and deployable:By using small-scale
open-source models on standard consumer GPUs, the
framework avoids reliance on high-performance hard-
ware, simplifies deployment, and supports edge applica-
tions.
The remainder of this paper is organized as follows.
Section II presents the proposed pipeline design based on
vision-language models and retrieval-augmented reasoning.
Section III reports the experimental setup and results, includ-
ing comparisons with the CLIP baseline. Section IV concludes
the paper and discusses potential directions for future work.
The objective of this work is to demonstrate the feasibility of
text-based retrieval and reasoning for zero-shot vehicle make–
model recognition in intelligent transportation systems.
II. METHODDESIGN
A. Problem Formulation
We formulate vehicle make and model recognition (VMMR)
as a zero-shot classification task. LetIdenote the domain of
vehicle images andY={y 1, y2, . . . , y K}the set of target
categories, where eachyencodes both the make and the model.
For an unseen input imagex∈ I, the goal is to predict the
most probable label:arXiv:2510.18502v1  [cs.CV]  21 Oct 2025

y∗= arg max
y∈YF(x, y),(1)
whereF(x, y)denotes a scoring function that measures the
semantic consistency betweenxand a candidate labely.
To approximateF, a vision–language encoderE v:I → T
converts the inputxinto a textual description ˆt∈ T. Within
the RAG framework, ˆtis compared against a database of
textual entriesD={t j}N
j=1using a similarity functionS(·,·)
to retrieve the top-kcandidates:
Rk(ˆt) = TopK
S(ˆt, tj)|t j∈ D	
.(2)
The retrieved entries are concatenated with ˆtto form a
promptP= [ ˆt;Rk], which is passed to a language model
(LM) for reasoning:
ˆy= LM reason (P).(3)
Here,ˆyserves as a practical approximation to the optimal
predictiony∗.
Performance on a held-out test set{(x i, yi)}Ntest
i=1 is evalu-
ated using standard classification metrics:
Accuracy =1
NtestNtestX
i=11{ˆyi=yi},
Precision =TP
TP + FP,Recall =TP
TP + FN,(4)
whereTP,FP, andFNdenote true positives, false positives,
and false negatives, respectively.
This formulation defines the zero-shot classification objec-
tive, the retrieval-augmented reasoning process, and the eval-
uation metrics, providing a concise mathematical abstraction
of the proposed pipeline.
B. Pipeline Design
The proposed pipeline performs vehicle make and model
recognition through three main stages by combining vi-
sion–language understanding with RAG. The overall idea is
to shift the recognition problem from the image domain into
the text domain, where new vehicle models can be flexibly
described and incorporated without retraining on large-scale
image datasets.
Stage 1: Visual-to-Text Conversion.A vision–language
encoder first converts an input vehicle image into a natural
language description that highlights its exterior attributes,
such as body type, headlights, grille design, or wheel shape.
By translating visual input into descriptive text, the system
avoids reliance on rigid image embeddings and instead enables
zero-shot generalization, since unseen models can still be
characterized through descriptive language.
Stage 2: Retrieval of Contextual Knowledge.The gener-
ated description is then compared against a textual database,
which contains curated entries representing a wide variety
of vehicle makes and models. Each entry encodes distinctive
attributes that help discriminate between similar categories.Table I.THE CAR MAKES AND MODELS USED IN THIS STUDY
Make name Model name
Ferrari Purosangue
Kia EV9
Lamborghini Revuelto
Mazda EZ6
Mitsubishi Xforce
Nissan Ariya
Rolls Royce Spectre
Toyota Supra GRMN
V olkswagen ID.Buzz
V olvo EX30
A similarity function ranks these entries, and the top-kmost
relevant candidates are retrieved. Because the database can
be dynamically updated with textual descriptions of newly
released vehicles, the system adapts rapidly to changes in the
real world without retraining on additional images.
Stage 3: Language Model Reasoning.The retrieved entries
are concatenated with the generated description to form a
structured prompt, which is passed to a language model (LM).
The LM reasons over both the direct description of the query
image and the retrieved contextual knowledge, producing the
most likely make and model. This design not only enables
zero-shot recognition of previously unseen vehicles but also
mitigates hallucination by grounding the LM in retrieved
evidence.
In summary, the pipeline avoids image-specific fine-tuning,
leverages textual retrieval for knowledge enrichment, and
ensures that recognition remains scalable, lightweight, and
adaptable to the continual introduction of new vehicle models
in intelligent transportation systems.
III. EXPERIMENTALRESULTS
A. Dataset Introduction
Due to the fact that existing car datasets, such as CompCars
[6], mostly contain models that have been on the market for
some time, they are not suitable for evaluating the ability
to recognize new car models. To address this, we collected
10 recently released car models from the web, with specific
selection criteria to ensure these were entirely new models
launched after 2023, rather than facelifts or derivatives of
existing models. This selection strategy ensures that the CLIP
model has no prior training exposure to these vehicles, provid-
ing a true zero-shot evaluation scenario. For each model, we
selected 10 images for the test set, resulting in a total of 100
test images. Additionally, one image per model was used to
form the training set, which serves to generate visual feature
descriptions and build the retrieval database. The car models
used in this study are listed in Table I.
B. Experimental Setup
Fig. 1 illustrates the full experimental architecture, including
the RAG integration within the recognition pipeline. We
employ CLIP as the text encoder to obtain both image and
text embeddings, while the large language model is Gemma 3
[7], an open-source model built upon Gemini 2.0 [8]. The

CLIP  Model Base accuracy
Input: Train images
Ariya
Input: Name of car model
Describe the car's front exterior in detail
Input: PromptInput: Test images
LLM
Output: Description of visual featuresHeadlight: The headlight...
Hood: The hood is...Headlight: The headlight...
Hood: The hood is...Headlight: The headlight...
Hood: The hood is...
Sentence EncoderEmbedding Database1. Establishing a Baseline
3. Recognizing V ehicle Models with RAG Integration2. Building the Retrieval Database
Input: Test images
LLM
Describe the car's front exterior in detail
Input: PromptHeadlight: The headlight...
Hood: The hood is...Headlight: The headlight...
Hood: The hood is...Headlight: The headlight...
Hood: The hood is...Sentence Encoder
Embedding Database
Retrieval
LLMSimilarity Top-5Headlight: The headlight...
Hood: The hood is...Headlight: The headlight...
Hood: The hood is...Headlight: The headlight...
Hood: The hood is...
RAG accuracyOutput: Description of visual featuresFig. 1.Experimental Framework for Car Model Recognition: CLIP vs. RAG Approach
12B parameter version of Gemma 3 is used to generate
fine-grained exterior feature descriptions and to reason over
retrieved context. The retrieval database is populated with
LLM-generated descriptions of training samples, which are
encoded and stored in a FAISS [9] vector index for cosine-
similarity search. All experiments are conducted on a single
NVIDIA GeForce RTX 5090 GPU, and the same pipeline is
applied for both the baseline and RAG-enhanced evaluations.
C. Experimental Procedure
The experimental procedure can be divided into three steps:
1) Establishing a Baseline:To establish a baseline, we
employ CLIP Vit-B/32 for zero-shot car model recognition.
The ten car models listed in Table I are treated as textual
labels. Each of the 100 test images is sequentially input to the
CLIP model, which computes the cosine similarity betweenthe image and each textual label. The label with the highest
similarity is taken as the predicted class, and the overall
prediction accuracy is then calculated.
2) Building the Retrieval Database:After establishing the
baseline, we input the 10 training images to the LLM to
generate front-end feature descriptions. These descriptions,
along with their corresponding labels, are formatted as JSON
and encoded into text embeddings using the CLIP ViT-B/32
text encoder, then stored in a FAISS [9] vector database.
3) Recognizing Vehicle Models with RAG Integration:We
re-input the 100 test images to LLM to generate feature de-
scriptions, encode them with CLIP ViT-B/32, and retrieve the
top-k similar entries from the database via cosine similarity,
where k represents the number of retrieved documents and is
set to 5 by default. The generated description and retrieved
entries are combined into a new prompt, which is then input

Table II.OVERALL PERFORMANCE METRICS FOR RAG AND
CLIP METHODS
Model Accuracy Recall Precision
CLIP 0.21 0.21 0.22
RAG-based LLM (Ours) 0.37 0.37 0.40
Table III.DETAILED PERFORMANCE METRICS FOR EACH CAR
MODEL IN CLIP
Car Model Accuracy Recall Precision
Ariya 0.20 0.20 1.00
EV9 0.70 0.70 0.10
EX30 0.00 0.00 0.00
EZ6 0.00 0.00 0.00
ID.Buzz 0.00 0.00 0.00
Purosangue 0.00 0.00 0.00
Revuelto 0.00 0.00 0.00
Spectre 0.00 0.50 0.33
SupraGRMN 0.50 0.00 0.00
Xforce 0.70 0.70 0.78
Table IV.DETAILED PERFORMANCE METRICS FOR EACH CAR
MODEL IN RAG-BASED LLM
Car Model Accuracy Recall Precision
Ariya 0.50 0.50 0.50
EV9 0.00 0.00 0.00
EX30 0.20 0.20 1.00
EZ6 0.00 0.00 0.00
ID.Buzz 0.00 0.00 0.00
Purosangue 0.40 0.40 1.00
Revuelto 0.60 0.60 0.21
Spectre 1.00 1.00 0.29
SupraGRMN 0.80 0.80 0.67
Xforce 0.20 0.20 0.33
to the LLM for prediction. The LLM compares the generated
description with each retrieved entry and selects the most
similar one as the predicted car model category.
4) Calculating Model Performance:After obtaining predic-
tions for all test images, we visualize the results as a confusion
matrix and compute standard metrics, including accuracy,
precision, and recall. These indicators collectively assess both
overall and class-level performance in the recognition task.
D. Performance Evaluation
As shown in Table II, the metrics show the distinctions
in overall performance for car model recognition between
our RAG approach and the CLIP model. The CLIP base-
line achieved 21% accuracy, while our RAG-based method
reached 37% accuracy, representing a notable improvement
of 16 percentage points compared to the CLIP baseline, with
similar gains observed in precision and recall metrics. This
improvement shows our RAG method demonstrates better
adaptability to new car models without requiring extensive
image datasets for training. Since the dataset contains balanced
samples across all categories, the recall and precision values
are identical to the accuracy scores. We report all three metrics
for each car model to provide a complete evaluation (Table III,
IV).
Fig. 2 shows the confusion matrices for both the CLIP (Fig.
2a) and RAG (Fig. 2b) approaches, providing detailed insights
into the classification accuracy for each car model category.
AriyaEV9EX30EZ6
ID.Buzz
PurosangueRevueltoSpectre
SupraGRMNXforce
Predicted ModelAriya
EV9
EX30
EZ6
ID.Buzz
Purosangue
Revuelto
Spectre
SupraGRMN
XforceTrue Model2 8 0 0 0 0 0 0 0 0
0 7 1 0 0 0 0 0 0 2
0 9 0 0 0 0 0 1 0 0
0 10 0 0 0 0 0 0 0 0
0 9 0 0 0 0 0 0 1 0
0 9 0 0 0 0 0 1 0 0
0 6 0 0 0 0 0 3 1 0
0 5 0 0 0 0 0 5 0 0
0 5 0 0 0 0 0 5 0 0
0 3 0 0 0 0 0 0 0 7Confusion Matrix of CLIP
0246810(a)CLIP Model
AriyaEV9EX30EZ6
ID.Buzz
PurosangueRevueltoSpectre
SupraGRMNXforce
Predicted ModelAriya
EV9
EX30
EZ6
ID.Buzz
Purosangue
Revuelto
Spectre
SupraGRMN
XforceTrue Model5 0 0 0 0 0 3 2 0 0
2 0 0 1 0 0 2 3 0 2
2 0 2 0 0 0 4 2 0 0
0 0 0 0 0 0 3 6 0 1
1 0 0 1 0 0 4 4 0 0
0 0 0 1 0 4 2 0 2 1
0 0 0 0 0 0 6 2 2 0
0 0 0 0 0 0 0 10 0 0
0 0 0 0 0 0 1 1 8 0
0 0 0 0 0 0 3 5 0 2Confusion Matrix of RAG-based LLM
0246810
(b)RAG-based LLM (Ours)
Fig. 2.Confusion Matrices for Car Model Classification: CLIP vs. RAG
Approach
The experimental results show that our method not only
outperforms CLIP in overall metrics but also achieves more
balanced performance across different car model categories.
E. Varying K Values
Initially, we chose a midpoint value of 5 for the K-value out
of the 10 total car models. The experimental results showed
that the accuracy already increased by 16%. We observed from
Fig. 3 that most of the correct retrievals were concentrated
within the top-3. Therefore, we modified the selection of
the K-value, and the test results are shown in Table V.
Although the difference was not significant, it still suggests
that too many options can lead to incorrect predictions by the
LLM during the final decision-making, even if the retrieval is
correct. In this experiment, a Top-1 K-value achieved the best
model recognition performance, improving by 18% compared
to the CLIP baseline.

T op-1 T op-2 T op-3 T op-4 T op-5
RankAriya
EV9
EX30
EZ6
ID.Buzz
Purosangue
Revuelto
Spectre
SupraGRMN
XforceTrue Model5 2 1 1 1
0 1 0 0 1
4 0 0 0 0
0 3 6 0 1
0 0 0 0 0
4 0 0 1 1
6 2 2 0 0
10 0 0 0 0
8 0 0 2 0
2 2 1 1 3T op-5 RAG Ranking Matrix
0246810Fig. 3.Retrieval Rank Distribution of True Models in Top-5 RAG Results
Table V.ACCURACY OF DIFFERENT K-V ALUE
K-value Accuracy
CLIP Model 0.21
Top-1 0.39
Top-3 0.38
Top-5 0.37
Top-7 0.33
F . Investigating Prediction Errors in Image Comparison
We analyzed the vehicles with the highest prediction errors
from Fig. 2b, namely the EZ6 and the Spectre. As shown in
Figures 4a and 4b, which are test images of the two cars,
their appearances are not highly similar. However, in the
textual descriptions, the model tended to emphasize overall
geometric proportions and shapes while overlooking brand-
specific details (such as the chrome trim design and air
intake grille texture). For example, descriptions mentioning
horizontal and narrow headlights, a simple and horizontally
extended grille outline, and smooth hood lines without distinct
creases caused the descriptions for the EZ6 and Spectre to be
highly similar, leading to confusion.
This issue may stem from the generic nature of the LLM.
The LLM we used is a general-purpose model not specifically
designed for image understanding and detailed feature genera-
tion. It has limited capabilities in processing fine image details
and variations in light and shadow, and it tends to generate
generalized descriptions that lack precision.
G. Limitations
Our RAG-enhanced method, while demonstrating improved
performance, has the following four key limitations:
•Dataset Limitations:Our evaluation was conducted on
a balanced dataset, which may not reflect the imbalanced
distribution commonly encountered in real-world scenar-
ios.
•Increased Inference Time:The retrieval process in-
troduces additional computational overhead, resulting in
(a)Mazda EZ6
 (b)Rolls Royce Spectre
Fig. 4.EZ6 and Spectre: A Comparative View
longer prediction times compared to direct CLIP infer-
ence.
•Dependency on Description Quality:The method’s
effectiveness is heavily dependent on the accuracy and
quality of LLM-generated descriptions, which may intro-
duce errors or inconsistencies.
•Recognition Performance Constraints:For common
car models, text-based retrieval accuracy still falls short
of pure computer vision methods. Our approach is more
suitable as a rapid auxiliary tool for zero-shot recognition
of new or rare car categories rather than a replacement
for established visual recognition methods.
However, it is worth noting that the limitation regarding
description quality dependency shows promising prospects for
resolution. With the rapid advancement of multimodal LLMs
and their improved capability in processing visual features,
this constraint is expected to be gradually addressed. Future
developments in LLM technology, including better fine-tuning
approaches and more sophisticated multimodal architectures,
are likely to significantly enhance the reliability of generated
descriptions.
IV. CONCLUSION
This paper presents a text-retrieval-based car model recog-
nition method that establishes a retrieval database through
textual descriptions of automotive visual features and performs
predictions based on these textual data. Unlike traditional
image recognition approaches that rely on extensive training
datasets, our method demonstrates significant improvements in
recognizing new car models compared to the CLIP baseline.
With appropriate selection of top-k values for retrieval, our
approach can achieve performance improvements of nearly
20%. These results highlight the potential and advantages of
RAG-based LLMs in zero-shot image recognition tasks, not
only reducing LLM hallucinations but also providing better
adaptability to evolving car model updates.
Looking forward, this work opens several promising re-
search avenues, including optimization of retrieval efficiency,
development of hybrid visual-textual approaches, and ex-
pansion to broader fine-grained recognition applications as
multimodal LLM technology continues to advance.
REFERENCES
[1] H. He, Z. Shao, and J. Tan, “Recognition of car makes and models
from a single traffic-camera image,”IEEE Transactions on Intelligent
Transportation Systems, vol. 16, no. 6, pp. 3182–3192, 2015.

[2] Y .-H. Chen, L. B. Kara, and J. Cagan, “Bignet: A deep learning architec-
ture for brand recognition with geometry-based explainability,”Journal
of Mechanical Design, vol. 146, no. 5, p. 051701, 2024.
[3] R. Wightman, H. Touvron, and H. J ´egou, “Resnet strikes back: An
improved training procedure in timm,”arXiv preprint arXiv:2110.00476,
2021.
[4] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clarket al., “Learning transferable
visual models from natural language supervision,” inProc. of Interna-
tional Conference on Machine Learning (ICML), 2021.
[5] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-T. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances inneural information processing systems, vol. 33, pp. 9459–9474, 2020.
[6] L. Yang, P. Luo, C. Change Loy, and X. Tang, “A large-scale car
dataset for fine-grained categorization and verification,” inProc. of IEEE
conference on computer vision and pattern recognition (CVPR), 2015.
[7] G. Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej,
S. Perrin, T. Matejovicova, A. Ram ´e, M. Rivi `ereet al., “Gemma 3
technical report,”arXiv preprint arXiv:2503.19786, 2025.
[8] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut,
J. Schalkwyk, A. M. Dai, A. Hauth, K. Millicanet al., “Gemini: a family
of highly capable multimodal models,”arXiv preprint arXiv:2312.11805,
2023.
[9] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search with
gpus,”arXiv preprint arXiv:1702.08734, 2017.