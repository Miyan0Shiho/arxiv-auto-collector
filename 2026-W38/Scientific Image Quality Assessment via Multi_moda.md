# Scientific Image Quality Assessment via Multi-modal Retrieval-Augmented Generation

**Authors**: Yinuo Zhang, Bingshuo Liu, Zhiying Tu, Dianhui Chu, Qingbin Liu, Xi Chen, Jiang Bian, Xiaoyan Yu, Dianbo Sui

**Published**: 2026-09-17 03:27:17

**PDF URL**: [https://arxiv.org/pdf/2609.19634v1](https://arxiv.org/pdf/2609.19634v1)

## Abstract
This paper proposes a Retrieval-Augmented Generation (RAG) framework for scientific image quality assessment, designed to simultaneously address both the understanding track (SIQA-U) and the scoring track (SIQA-S) of the SIQA challenge. We construct a multimodal index that integrates textual semantics with fine-grained visual features, and develop a multi-route retrieval and fusion mechanism to provide large language models with highly relevant reference cases, thereby enhancing their capability to evaluate complex scientific images. Experimental results demonstrate that the proposed framework effectively aligns with the judgment criteria of human experts. Ultimately, our method achieves 1st place in the SIQA-U track of the SIQA challenge at the ICME 2026 Grand Challenges.

## Full Text


<!-- PDF content starts -->

Scientific Image Quality Assessment via
Multi-modal Retrieval-Augmented Generation
Yinuo Zhang1, Bingshuo Liu1, Zhiying Tu1, Dianhui Chu1,
Qingbin Liu2, Xi Chen2, Jiang Bian2, Xiaoyan Yu3, Dianbo Sui1
DBTeam
1Harbin Institute of Technology2Tencent3Nanyang Technological University
yinuo@stu.hit.edu.cn, suidianbo@hit.edu.cn
Abstract—This paper proposes a Retrieval-Augmented Gener-
ation (RAG) framework for scientific image quality assessment,
designed to simultaneously address both the understanding track
(SIQA-U) and the scoring track (SIQA-S) of the SIQA challenge.
We construct a multimodal index that integrates textual seman-
tics with fine-grained visual features, and develop a multi-route
retrieval and fusion mechanism to provide large language models
with highly relevant reference cases, thereby enhancing their
capability to evaluate complex scientific images. Experimental
results demonstrate that the proposed framework effectively
aligns with the judgment criteria of human experts. Ultimately,
our method achieves 1st place in the SIQA-U track of the SIQA
challenge at the ICME 2026 Grand Challenges1.
I. INTRODUCTION
In recent years, with the rapid development of Multimodal
Large Language Models (MLLMs), these models have demon-
strated remarkable capabilities in image understanding and
cross-modal reasoning tasks [1]. In traditional Image Quality
Assessment (IQA), MLLMs are already capable of producing
relatively accurate quality judgments for natural images by
leveraging both visual features and semantic information [2],
[3]. However, scientific images—such as molecular structure
diagrams, experimental workflow charts, and geometric illus-
trations—are inherently structured representations of domain
knowledge. Their quality depends not only on visual clarity but
also on the correctness of scientific logic and the rigor of their
presentation. Consequently, an image that appears visually
clear and aesthetically pleasing may still be scientifically
unacceptable due to factual inaccuracies or missing domain-
specific details, making scientific image quality assessment a
significantly more challenging problem [4].
Existing approaches typically rely on supervised fine-tuning
(SFT) [5] to learn a direct mapping from images to quality
labels. However, such methods remain limited in the SIQA-U
track [4]. The fundamental issue lies in the lack of explicit
modeling of how humans evaluate image quality. Although
models are capable of answering image-related questions, they
tend to focus on high-level semantic content during inference,
often overlooking subtle yet critical visual defects, which leads
to biased judgments. Therefore, relying solely on implicitly
encoded knowledge in model parameters is insufficient for
stable and reliable scientific image quality assessment.
Corresponding author: Dianbo Sui.
1https://2026.ieeeicme.org/grand-challenges/To address these challenges, we propose a Retrieval-
Augmented Generation (RAG) [6] framework for scientific im-
age quality assessment. The core idea is to introduce reference
cases that are highly relevant to the current sample in both
semantic and visual spaces, thereby transforming evaluation
criteria that are implicitly stored in model parameters into
explicit contextual information. This enables a more direct
alignment between question semantics and critical local details
in the image, ultimately improving the model’s ability to detect
subtle but important defects.
Specifically, we first construct a multimodal index that
integrates textual semantics with fine-grained visual features.
On the textual side, semantic embeddings are used to model
questions and task attributes, while on the visual side, patch-
level representations are employed to preserve local image
details. Based on this, we design a multi-route retrieval mech-
anism tailored to different question types. For understanding-
oriented tasks, we combine text-based semantic retrieval with
text-guided visual retrieval to capture the correspondence
between question semantics and local image features. For
overall quality assessment tasks, we further incorporate image-
to-image visual retrieval to obtain structurally similar reference
images as quality anchors. The results from multiple retrieval
routes are unified through weighted fusion and rule-based
re-ranking, enabling the selection of the most informative
reference cases.
During inference, the retrieved high-relevance cases are
jointly fed with the test sample into a large language model,
guiding it to perform reasoning or scoring in a case-driven
manner. The proposed framework significantly enhances the
model’s ability to distinguish fine-grained defects and knowl-
edge errors in the SIQA-U track, while achieving strong align-
ment with human subjective ratings in the SIQA-S track. These
results validate the effectiveness of the retrieval-augmented
multimodal reasoning paradigm for scientific image quality
assessment.
II. DATAPREPARATION
To support RAG-based scientific image quality assessment,
we construct a multimodal index that integrates textual seman-
tics with visual information.
arXiv:2609.19634v1  [cs.CV]  17 Sep 2026

TABLE I
COMPLETE SAMPLE STRUCTURE OF THESIQA-UTRAINING SET
Field Description
type Question type (yes-or-no / what / how)
category Evaluation dimension
question Question text describing the visual assessment task
option Multiple-choice options
answer Correct option label (A/B/C/D)
explanation Expert explanation
image Path to the corresponding scientific image
A. SIQA-U
The SIQA-U training set contains 104,000 samples, while
the validation and test sets each consist of 1,120 samples2.
In addition to images, each sample is associated with rich
semantic information, as summarized in Table I. Notably,how-
type questions in the training set contain only two options,
whereas those in the validation and test sets provide four
options, leading to an inconsistency in option structure. To
ensure alignment between retrieved reference cases and test
samples, we construct the index foryes-or-noandwhat
questions based on the training set, whilehow-type questions
are indexed separately using the validation set.
On this basis, we build both textual semantic and visual se-
mantic indices to retrieve reference cases from complementary
perspectives.
a) Textual Semantic Index.:For each sample, we con-
catenate theclass,category, andquestionfields into a de-
scriptive text sequence. This sequence is then encoded into
a high-dimensional dense vector using thetext-embedding-3-
largemodel [7] and stored in the Qdrant3vector database.
Each vector record is augmented with metadata, including
the ground-truth answer, expert explanation, and image path,
enabling the model to leverage these references during in-
ference. Although validation samples do not contain expert
explanations, their answer options themselves include detailed
textual descriptions of different quality levels, which are
directly used as criteria for quality assessment in downstream
reasoning.
b) Visual Semantic Index.:Scientific image quality as-
sessment often depends on fine-grained visual details dis-
tributed across the image (e.g., axis labels, legends, and
scale bars), which cannot be adequately captured by a single
global representation. To address this, we employ ColPali [8]
(based on PaliGemma-3B [9]) to encode images from both
the training and validation sets. This approach partitions
each image into multiple local patches and generates a 128-
dimensional vector for each patch, resulting in approximately
1,000 vectors per image. Such a multi-vector representation
preserves local structural features at a fine-grained level,
rather than compressing them into a single global embedding.
Compared to conventional visual encoders such as CLIP [10],
ColPali demonstrates stronger retrieval capability in scenarios
2https://huggingface.co/datasets/SIQA/TrainSet
3https://github.com/qdrant/qdrantinvolving document-style and scientific images with complex
local structures. All visual vectors are also stored in Qdrant,
jointly forming a dual-route retrieval foundation with the
textual semantic index.
B. SIQA-S
In the SIQA-S training set, each sample contains only the
image path and two continuous scores (perception rating and
knowledge rating), without any accompanying textual descrip-
tion. Therefore, only a visual semantic index is constructed for
this track.
Specifically, we adopt the same encoding strategy as in
SIQA-U by applying ColPali to generate patch-based multi-
vector representations for all training images, which are then
stored in Qdrant. The corresponding perception rating and
knowledge rating are stored as metadata alongside the vectors,
serving as reference anchors during retrieval and supporting
subsequent quality prediction.
III. METHODOLOGY
The overall framework of this study is illustrated in Fig. 1.
The proposed method is built upon a Multimodal RAG
paradigm. For each test sample, the system first retrieves
highly relevant reference cases from a pre-built multimodal
index in both semantic and visual spaces, and then feeds these
cases together with the test sample into a large language model
to support downstream reasoning and prediction.
A. SIQA-U
a) Hard Filtering:To reduce cross-domain noise, we first
apply a hard filtering step before vector retrieval. Specifically,
candidate samples are restricted to the same scientific domain
as the test sample according to theclassattribute, thereby
improving retrieval relevance.
b) Multi-Route Retrieval:Foryes-or-noandwhatques-
tions, we perform dual-route retrieval:
Textual Semantic Retrieval.We concatenate theclass,
category, andquestionfields of the query sample into a single
textual input and encode it into a dense vectorqusingtext-
embedding-3-large. The cosine similarity between the query
vectorqand each indexed text vectork iis computed as:
stext(i) =q·k i
∥q∥∥k i∥(1)
A higher score indicates greater semantic similarity between
the candidate and the query.
Text-driven Visual Retrieval.We further encode the ques-
tion text using the ColPali text encoder to obtain a set of
patch-level query embeddingsQ={q 1,q2, . . . ,q m}. Each
candidate image is represented by a set of patch embeddings
Di={d 1,d2, . . . ,d n}. We compute the MaxSim similarity
as:
scolpali(i) =X
qj∈Qmax
dk∈D icos(q j,dk)(2)
This mechanism aligns textual semantics with fine-grained
visual patterns. For example, when a question involves the

Index Construction
Text Semantic Index Visual Semantic Index
class + category + question
text-embedding-3-large
...
Metadata
answer
explanation
option
class
category
question
typeMetadata
answer
explanation
option
class
category
question
typeQdrant
（Text Index）ColPali
image n-patch
...
Qdrant
（Visual Index ）Retrieval & Fusion Pipeline
test imageclass Mathematical Representation
category knowledge correctness
type how
questionHow accurately does the image represent a cyclic
quadrilateral according to geometric principles?
option A ...     B...   C...   D...
Hard Filtering by class
Multi-route Retrieval
Text Semantic Retrieval Text-driven Visual Retrieval Image2Image Retrieval
Query: class + category + question
text-embedding-3-large
...
Cosine Similarity
Top-KQuery: Text
ColPail Text Encoder
...
...
MaxSim
Top-KQuery: Image
ColPail Image Encoder
MaxSim
Top-K
RRF Fusion
Re-ranking
1
2
3
4category & type match
category match only
type match only
neither matchTop-5 Selection
Prompt Construction & Inference
InputReference Case #5
Reference Case #4
Reference Case #3
Reference Case #2
Reference Case #1（Least relevant ）
（Least relevant ）
Test Question
Question: How accurately
does the image represent ...
Option: A ...   B ...   C ...   D ...
OutputAnswer: X+
GPT - 5.4Fig. 1. Overview of the Multi-modal Retrieval-Augmented Generation Framework for SIQA-U.
correctness of geospatial annotations, the model can prioritize
retrieving images containing similar visual elements such as
compass indicators or coordinate systems, thereby providing
more targeted visual references.
Forhow-type questions, we additionally introduce image-
to-image retrieval. The test image is encoded into patch-level
embeddingsQimg={qimg
1,qimg
2, . . . ,qimg
m}, and similarity is
computed against candidate images as:
simg2img (i) =X
qimg
j∈Qimgmax
dk∈D icos(qimg
j,dk)(3)
Sincehow-type questions primarily require holistic quality
judgment, structurally similar reference images serve as strong
quality anchors. Therefore, this retrieval route is assigned a
higher weight during fusion.
c) RRF Fusion and Reranking:The results from multiple
retrieval routes are aggregated using weighted Reciprocal RankFusion (RRF) [11]:
sRRF(i) =RX
r=1wr
k+rank r(i)(4)
whereRdenotes the number of retrieval routes, rank r(i)is the
rank of sampleiin router,k= 60is a smoothing parameter,
andw ris the corresponding weight. Foryes-or-noandwhat
questions, both routes are assigned equal weights (w r= 1.0).
Forhow-type questions, the image-to-image route is assigned
a higher weight of 2.0, while other routes are set to 1.0.
After fusion, we further apply a rule-based reranking strat-
egy based on the consistency ofcategoryandtypewith the
test sample. Candidates are divided into four priority levels:
(1) both category and type match, (2) category match only,
(3) type match only, and (4) neither match. Within each
level, samples are sorted by their RRF scores. The final top-
5 references are selected by filling slots from higher to lower
priority levels sequentially, ensuring that the most task-relevant
cases are always included.

d) Prompt Construction and Inference:The selected top-
5 reference cases are ordered from least to most relevant, with
the most relevant case placed last to exploit the recency bias
of large language models. Each case includes the reference
image,category,question,option, andexplanation(if avail-
able). The test image and question are appended at the end.
The model is instructed to perform step-by-step reasoning and
output the final answer in the format “Answer: X”, where
X∈ {A, B, C, D}. We use GPT-5.4 [12] as the backbone
model with temperature set to 0.
B. SIQA-S
The SIQA-S task requires predicting continuous scores for
both perceptual quality and knowledge quality. Since each
training sample only contains image paths and corresponding
scores without any textual annotations, we rely solely on visual
retrieval.
a) Hard Filtering:Similar to SIQA-U, candidate sam-
ples are first filtered based on theirclassattribute to ensure
domain consistency.
b) Visual Retrieval:For each test image, image-to-image
retrieval is performed using ColPali-based patch embeddings.
The similarity is computed as:
simg2img (i) =X
qimg
j∈Qimgmax
dk∈D icos(qimg
j,dk)(5)
The top-5 most visually similar training samples are selected
as references.
c) Prompt Construction and Inference:The retrieved
reference images and their correspondingperception rating
andknowledge ratingare incorporated into the prompt as
quality anchors. To ensure comprehensive evaluation, the
model simultaneously predicts both perceptual and knowledge
quality scores in a single inference pass. The output format
is “perception: X.XX / knowledge: X.XX”, where scores are
constrained to the range[1.00,5.00].
IV. PLATFORM& KEYCASES
A. Experimental Platform
All experiments in this study were conducted on a server
equipped with a single NVIDIA GeForce RTX 4090 (24GB).
The software environment was based on Python 3.10 and Py-
Torch 2.6.0 [13]. Vector index construction and online retrieval
were implemented using the Qdrant 1.13.3 vector database,
and GPT-5.4 was employed as the core large language model
during the inference stage.
B. Experimental Results
We compared the proposed method with other top-tier teams
in the challenge and the official baseline. Table II and Table III
present the final official leaderboards for the Understanding
track (SIQA-U) and the Scoring track (SIQA-S), respectively.
Our method (DB Team) achieved the first-place ranking in
the SIQA-U track. While competition was intense on relatively
simple closed-ended questions like Yes/No and What, ourTABLE II
SIQA-U LEADERBOARD
Rank TeamYes/No
ACC (%)What
ACC (%)How
ACC (%)Final
1 DB Team (Ours)53.95 78.8634.87 51.88
2 Tomcat58.68 83.71 28.21 50.95
3 leitinggaba 53.95 81.43 31.28 50.86
4 DoubleY 55.79 75.43 27.18 47.38
5 YWJC 46.05 70.29 28.21 44.40
6 Baseline 44.47 62.57 30.26 42.79
TABLE III
SIQA-S LEADERBOARD
Rank TeamPerception
(SRCC / PLCC)Knowledge
(SRCC / PLCC)Final
1 Tomcat0.9135 / 0.9253 0.9282 / 0.9527 92.99
2 DoubleY 0.9069 / 0.9271 0.9130 / 0.9414 92.21
3 leitinggaba 0.9031 / 0.9205 0.9038 / 0.9379 91.63
4 YWJC 0.8915 / 0.9133 0.8824 / 0.9147 90.05
5 NJU SIQA Team 0.8763 / 0.9071 0.8514 / 0.8948 88.24
6 DB Team (Ours)0.8352 / 0.8647 0.9097 / 0.9065 87.90
7 CSQA 0.8462 / 0.8734 0.8571 / 0.8860 86.57
8 Baseline 0.7071 / 0.7068 0.8366 / 0.8521 77.57
framework demonstrated exceptional performance on How-
type questions—which involve complex quality attribution and
deep reasoning—attaining an accuracy of 34.87%, signifi-
cantly surpassing all other competing teams and the baseline
model. This success is primarily attributed to our dual-route
retrieval fusion mechanism, which ensures that the model
can not only capture macro-level semantics but also precisely
locate critical local visual details in complex scientific images.
Compared to the official baseline (gpt-4o), our framework
achieved substantial improvements across all metrics, with a
notable gain of approximately 9.1 points in the SIQA-U final
score and 10.3 points in SIQA-S. These results validate the
effectiveness of the proposed retrieval-augmented multimodal
reasoning paradigm for scientific image quality assessment.
C. Qualitative Analysis of Key Cases
We conduct a qualitative analysis on several challenging
key test samples, primarily covering complex scenarios such
as subtle artifacts and ambiguous content. Since the ground-
truth labels for the test set are unavailable during the evaluation
phase, this section focuses on presenting the model’s predicted
scores and its reasoning chains to evaluate their alignment with
human perception and intuition.
Table IV illustrates the representative case studies for the
SIQA-S track, while Table V and Table VI supplements
the corresponding retrieval results that support the model’s
inference. The specific case analysis demonstrates that our
framework can effectively retrieve samples highly similar to
the test images to serve as scoring anchors. Based on these
anchors, the generated assessment results exhibit high consis-
tency with human experts overall. This further validates the
effectiveness of the proposed retrieval-augmented multimodal

TABLE IV
QUALITATIVE ANALYSIS OF REPRESENTATIVE CASES IN THESIQA-STRACK.
Case ID Perception Knowledge Reasoning and Analysis
105 3.05 2.99 Balances visual clarity with informational accuracy, noting moderate semantic ambiguity within
the image.
407 2.08 1.04 Identifies severe logical structural breaks; visual artifacts prevent accurate transmission of
scientific knowledge.
415 2.11 1.00 Captures high-frequency image noise, leading to the failure of knowledge verification due to
missing key visual features.
473 4.86 2.02 Extremely high rendering quality, but deep reasoning identifies a severe deviation between
diagram logic and scientific common sense.
485 2.64 2.79 Detects subtle conflicts between visual cues and text labels, reflecting the model’s sensitivity to
multimodal alignment.
625 3.99 3.99 Exhibits high inter-modal consistency with clear image expression and content meeting scientific
rigor requirements.
677 4.23 3.99 Acknowledges complete presentation of core scientific conclusions despite minimal layout
imperfections.
686 4.33 3.98 Effectively handles complex semantic backgrounds, verifying academic logical coherence while
maintaining high perceptual scores.
855 3.98 3.70 Identifies slight pixel distortion at the edges but determines it is insufficient to impair the overall
scientific information delivery.
1033 4.47 4.99 Accurately captures exceptional knowledge accuracy even when there is slight room for
improvement in perceptual quality.
reasoning framework in the task of scientific image quality
assessment.
V. CONCLUSION
This paper proposes a scientific image quality assessment
(SIQA) framework based on Multimodal Retrieval-Augmented
Generation (RAG), designed to unify the tasks of both SIQA-
U and SIQA-S tracks. By constructing a multimodal index li-
brary that integrates textual semantics with fine-grained visual
features, and designing mechanisms for multi-way retrieval,
weighted fusion, and case-driven reasoning, the model effec-
tively leverages highly relevant reference cases during the in-
ference process. This significantly enhances the discriminative
capability for subtle visual defects in scientific images. Our
method achieved first place in the SIQA-U track, validating the
effectiveness of the retrieval-augmented multimodal reasoning
paradigm in scientific image quality assessment tasks.
ACKNOWLEDGMENT
This work is supported by the National Key Re-
search and Development Program of China (Grant No.
2023YFB3307500), the National Natural Science Foundation
of China (Grant No. 62306087 and 62472121), the Natu-
ral Science Foundation of Shandong Province (Grant No.
ZR2023QF154), the Key Research and Development Pro-
gram of Shandong Province (Grant No. 2025CXPT077), the
Research on Cognitive Processing Technologies for Multi-
modal Big Data in Policing Information Project (Grant No.
2024DXZD0004), the Special Funding Program of Shandong
Taishan Scholars Project and CCF-Tencent Rhino-Bird Open
Research Fund (Grant No. CCF-Tencent RAGR20250105).REFERENCES
[1] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu,
and Enhong Chen, “A survey on multimodal large language models,”
National Science Review, vol. 11, no. 12, Nov. 2024.
[2] Tianhe Wu, Kede Ma, Jie Liang, Yujiu Yang, and Lei Zhang, “A
comprehensive study of multimodal large language models for image
quality assessment,” 2024.
[3] Jiebin Yan, Ziwen Tan, Yuming Fang, Jiale Rao, and Yifan Zuo,
“Max360iq: Blind omnidirectional image quality assessment with multi-
axis attention,” 2025.
[4] Wenzhe Li, Liang Chen, Junying Wang, Yijing Guo, Ye Shen, Farong
Wen, Chunyi Li, Zicheng Zhang, and Guangtao Zhai, “Siqa: Toward
reliable scientific image quality assessment,” 2026.
[5] Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun,
Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, and Guoyin
Wang, “Instruction tuning for large language models: A survey,” 2025.
[6] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi
Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang, “Retrieval-
augmented generation for large language models: A survey,” 2024.
[7] OpenAI, “New embedding models and api updates,” https://openai.com/
blog/new-embedding-models-and-api-updates, 2024, Accessed: 2026-
05-03.
[8] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud,
C´eline Hudelot, and Pierre Colombo, “Colpali: Efficient document
retrieval with vision language models,” 2025.
[9] Lucas Beyer, Andreas Steiner, Andr ´e Susano Pinto, Alexander
Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Al-
abdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Un-
terthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner,
Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian
Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bo ˇsnjak, Xi Chen,
Matthias Minderer, Paul V oigtlaender, Ioana Bica, Ivana Balazevic,
Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu
Soricut, Jeremiah Harmsen, and Xiaohua Zhai, “Paligemma: A versatile
3b vlm for transfer,” 2024.
[10] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel
Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin,
Jack Clark, Gretchen Krueger, and Ilya Sutskever, “Learning transferable
visual models from natural language supervision,” 2021.
[11] Gordon V . Cormack, Charles L. A. Clarke, and Stefan B ¨uttcher, “Re-
ciprocal rank fusion outperforms condorcet and individual rank learning

methods,”Proceedings of the 32nd international ACM SIGIR conference
on Research and development in information retrieval, 2009.
[12] OpenAI, “Gpt-5.4,” https://openai.com/zh-Hans-CN/index/
introducing-gpt-5-4/, 2026, Accessed: 2026-05-03.
[13] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Brad-
bury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein,
Luca Antiga, Alban Desmaison, Andreas K ¨opf, Edward Yang, Zach
DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit
Steiner, Lu Fang, Junjie Bai, and Soumith Chintala, “Pytorch: An
imperative style, high-performance deep learning library,” 2019.

APPENDIXA
PROMPTTEMPLATES
A. SIQA-U
SIQA-U System Prompt
You are an expert evaluator of scientific research figures. Your task is to answer
multiple-choice questions about the quality of scientific images. You will be given
several reference cases, followed by the test question. Study the reasoning patterns in
the reference cases carefully, then answer the test question. You MUST end your response
with ’Answer: X’ where X is a single uppercase letter.
SIQA-U User Prompt
--- Reference Case{i}---
Domain:{class}
Evaluation Dimension:{category}
Question:{question}
Options:{option}
Expert Reasoning:{explanation}
Correct Answer:{answer}
[base64 image]
--- Test Question ---
Evaluation Dimension:{category}
Question:{question}
Options:{option}
Please analyze this image based on the reasoning patterns shown in the reference cases
above. Think step by step, then end with ‘Answer: X’.
[base64 image]
B. SIQA-S
SIQA-S System Prompt
You are an expert evaluator of scientific research figures. Your task is to rate a test
image on two dimensions:
- perception_rating: subjective quality (sharpness, layout, aesthetics), score 1.00-5.00
- knowledge_rating: objective quality (scientific rigor, completeness, correctness),
score 1.00-5.00
You will be given reference images with their human ratings. Study them carefully and
use them as anchors to rate the test image.
You MUST output ONLY in this exact format (two decimal places):
perception: X.XX
knowledge: X.XX
SIQA-S User Prompt
--- Reference Case{i}(Domain:{class}) ---
perception_rating:{perception_rating}
knowledge_rating:{knowledge_rating}
[base64 image]
--- Test Image (Domain:{class}) ---
Please rate this image based on the reference cases above.
Output ONLY:
perception: X.XX
knowledge: X.XX
[base64 image]

TABLE V
S-TRACKRETRIEVALRESULTS FORSELECTEDTESTCASES(PART1)
ID Test Image Ref 1 Ref 2 Ref 3 Ref 4 Ref 5
105
perception: 3.00
knowledge: 2.33
perception: 3.33
knowledge: 2.00
perception: 4.00
knowledge: 2.67
perception: 3.33
knowledge: 3.00
perception: 3.67
knowledge: 3.00
407
perception: 2.00
knowledge: 1.00
perception: 2.00
knowledge: 1.33
perception: 2.33
knowledge: 1.33
perception: 2.00
knowledge: 1.00
perception: 2.33
knowledge: 1.33
415
perception: 2.33
knowledge: 1.33
perception: 2.67
knowledge: 1.00
 perception: 2.33
knowledge: 1.33
perception: 2.33
knowledge: 1.00
perception: 2.33
knowledge: 1.00
473
perception: 4.33
knowledge: 3.33
perception: 4.00
knowledge: 4.33
perception: 4.00
knowledge: 3.00
 perception: 2.67
knowledge: 1.33
perception: 3.33
knowledge: 2.00
485
perception: 2.00
knowledge: 3.00
perception: 2.00
knowledge: 3.67
perception: 2.00
knowledge: 3.00
perception: 2.00
knowledge: 3.67
perception: 2.67
knowledge: 3.67

TABLE VI
S-TRACKRETRIEVALRESULTS FORSELECTEDTESTCASES(PART2)
ID Test Image Ref 1 Ref 2 Ref 3 Ref 4 Ref 5
625
perception: 2.67
knowledge: 3.67
perception: 3.67
knowledge: 4.00
 perception: 4.00
knowledge: 3.67
perception: 3.33
knowledge: 3.00
 perception: 4.00
knowledge: 2.67
677
perception: 4.33
knowledge: 4.00
perception: 2.33
knowledge: 2.00
perception: 3.33
knowledge: 3.33
perception: 2.67
knowledge: 4.00
perception: 3.67
knowledge: 3.00
686
perception: 4.67
knowledge: 3.67
perception: 4.33
knowledge: 4.00
perception: 4.00
knowledge: 3.33
perception: 4.67
knowledge: 4.33
perception: 4.33
knowledge: 3.67
855
perception: 4.33
knowledge: 3.33
perception: 4.00
knowledge: 4.33
perception: 4.67
knowledge: 4.00
perception: 4.33
knowledge: 4.00
 perception: 3.33
knowledge: 4.00
1033
perception: 4.67
knowledge: 4.33
perception: 4.67
knowledge: 4.67
perception: 4.67
knowledge: 4.67
perception: 5.00
knowledge: 5.00
perception: 5.00
knowledge: 5.00