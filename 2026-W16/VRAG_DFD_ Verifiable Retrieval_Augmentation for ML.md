# VRAG-DFD: Verifiable Retrieval-Augmentation for MLLM-based Deepfake Detection

**Authors**: Hui Han, Shunli Wang, Yandan Zhao, Taiping Yao, Shouhong Ding

**Published**: 2026-04-15 09:27:49

**PDF URL**: [https://arxiv.org/pdf/2604.13660v2](https://arxiv.org/pdf/2604.13660v2)

## Abstract
In Deepfake Detection (DFD) tasks, researchers proposed two types of MLLM-based methods: complementary combination with small DFD detectors, or static forgery knowledge injection.The lack of professional forgery knowledge hinders the performance of these DFD-MLLMs.To solve this, we deeply considered two insightful issues: How to provide high-quality associated forgery knowledge for MLLMs? AND How to endow MLLMs with critical reasoning abilities given noisy reference information? Notably, we attempted to address above two questions with preliminary answers by leveraging the combination of Retrieval-Augmented Generation (RAG) and Reinforcement Learning (RL).Through RAG and RL techniques, we propose the VRAG-DFD framework with accurate dynamic forgery knowledge retrieval and powerful critical reasoning capabilities.Specifically, in terms of data, we constructed two datasets with RAG: Forensic Knowledge Database (FKD) for DFD knowledge annotation, and Forensic Chain-of-Thought Dataset (F-CoT), for critical CoT construction.In terms of model training, we adopt a three-stage training method (Alignment->SFT->GRPO) to gradually cultivate the critical reasoning ability of the MLLM.In terms of performance, VRAG-DFD achieved SOTA and competitive performance on DFD generalization testing.

## Full Text


<!-- PDF content starts -->

VRAG-DFD: Verifiable Retrieval-Augmentation for MLLM-based
Deepfake Detection
Hui Han1,2* *Shunli Wang2*Yandan Zhao2
Taiping Yao2Shouhong Ding2†
1Shanghai Jiao Tong University2Tencent Youtu Lab
https://github.com/abigcatcat/VRAG-DFD.git
Abstract
In Deepfake Detection (DFD) tasks, researchers
proposed two types of MLLM-based methods: com-
plementary combination with small DFD detectors, or
static forgery knowledge injection. The lack of pro-
fessional forgery knowledge hinders the performance
of these DFD-MLLMs. To solve this, we deeply con-
sidered two insightful issues: How to provide high-
quality associated forgery knowledge for MLLMs?
AND How to endow MLLMs with critical reason-
ing abilities given noisy reference information? No-
tably, we attempted to address above two questions
with preliminary answers by leveraging the combina-
tion of Retrieval-Augmented Generation (RAG) and
Reinforcement Learning (RL). Through RAG and RL
techniques, we propose theVRAG-DFDframework
with accurate dynamic forgery knowledge retrieval
and powerful critical reasoning capabilities. Specif-
ically, in terms of data, we constructed two datasets
with RAG: Forensic Knowledge Database (FKD) for
DFD knowledge annotation, and Forensic Chain-of-
Thought Dataset (F-CoT), for critical CoT construc-
tion. In terms of model training, we adopt a three-
stage training method (Alignment→SFT→GRPO) to
gradually cultivate the critical reasoning ability of
the MLLM. In terms of performance,VRAG-DFD
achieved SOTA and competitive performance on DFD
generalization testing.
1. Introduction
With the rapid advancement of Artificial Intelli-
gence Generated Content (AIGC) technology, Deep-
fake techniques [23, 46, 47, 55] (e.g., face swapping
and facial reenactment) and image generation meth-
ods [12, 16, 39] have already acquired the ability to
produce high-fidelity forged images. While such tech-
nologies demonstrate promising applications in enter-
tainment media, film production, and other fields, they
*Equal Contribution.
†Corresponding author (email: ericshding@tencent.com).
Prompt
Query Query
✗High hallucination
✗Reference- free
✗Static knowledge✓Low hallucination
✓Case reference
✓Dynamic knowledgekTop-k 
Retrieval Evidence
Detailed
Prompt
This is fake, because the 
mouth is blurry and the eyebrow texture is abnormal.Critical Reasoning CoT
❶<Preliminary Visual Analysis>
❷<RAG Reference Information Analysis>
❸<Fusion, Reasoning, and Decision>
❹<Answer>: Fake
Generator Generator
(a) Previous framework. (b) The proposed VRAG -DFD framework.Figure 1. Comparison between the previous method and
the proposedVRAG-DFDframework. Previous methods
predicted based on a single image and prompt with severe
hallucinations. The proposedVRAG-DFDeffectively re-
duce hallucinations by constructing highly relevant refer-
ences through RAG and making decisions through critical
reasoning. (Faces are selected from publicly available FF++
[40] dataset.)
also pose severe security risks, including privacy in-
fringement, financial fraud, and disinformation dis-
semination. As a core defensive measure, Deepfake
Detection (DFD) technology has grown increasingly
critical for ensuring the credibility of visual content in
the era of AIGC-driven forgery.
Traditional deepfake detectors [6, 11, 19, 21, 24,
41, 50, 59] rely on fine-tuning visual backbone net-
works (e.g., CNNs or ViTs) and enhance generaliz-
ability through delicate network designs. These face
forgery detectors typically aim to mine discrimina-
tive fake patterns (e.g., inconsistent facial textures, ab-
normal lighting) and thereby distinguish fake images
[50]. As simple binary classifiers, these detectors can
only output probabilistic results and cannot generate
interpretable text, thus failing to meet users’ interac-
tive needs, such as the demand for detailed analysisarXiv:2604.13660v2  [cs.CV]  16 Apr 2026

onWhy an image is classified as fake?[7].
The emergence of Multimodal Large Models
(MLLMs) [4] has offered new prospects for address-
ing the aforementioned challenges. Some existing
research [7, 13, 20, 34, 52] has already integrated
MLLMs into Deepfake detection tasks, enabling these
detectors to generate textual explanations and thus en-
hancing model interpretability. Some models [7, 20,
34] adopt a hybrid framework combining MLLM and
small DFD detectors, enabling a clear division of labor
betweenHigh-levelsemantic analysis andLow-level
artifacts extraction. Other models [13, 52] injected
DFD expertise through sparse static external knowl-
edge such as isolated keyword or attribute lists. We
argue that the combination strategy and static knowl-
edge injection approach cannot enable MLLM to have
true understanding, criticism, and thinking abilities
in DFD task, which means that theKnowledge Gap
problem still exists. To solve this, we need to address
two unavoidable issues:
Q1: How to provide high-quality detailed associ-
ated DFD knowledge for MLLM ?
Q2: How to endow MLLM with both critical rea-
soning abilities and the capacity to recognize the as-
sociated knowledge?
ForQ1, we adopt Retrieval-Augmented Gener-
ation (RAG) to solve the detailed associated DFD
knowledge issue. The existing annotation methods
[13, 52] usually start with designing a detailed an-
notation prompt, and then use closed or open source
LLMs to annotate individual fake images. Assum-
ing this process is denoted asP(T|M, I), whereM
denotes the annotator,Idenotes the single target im-
age,Tdenotes the annotation text. We believe that
a feasible solution to achieve better deepfake knowl-
edge is to provide a series ofKhighly relevant cases
for auxiliary annotation. The annotation process be-
comesP(T|M, I,{I k}K
1,{Tk}K
1). Specifically, we
constructed Forensic Knowledge Database (FKD) and
Forensic Chain-of-Thought (F-CoT) Datsets to obtain
more reliable associated DFD annotations.
ForQ2, we proposeVRAG-DFD, an interpretable
Deepfake detection framework based on RAG and
critical thinking training, as shown in Figure 1.
Specifically, based on the above reliable annotations,
we designed a three-stage training strategy to grad-
ually inject critical reasoning abilities into MLLMs.
(1) Firstly, we equip the MLLM with basic visual
perception capabilities through alignment learning on
large-scale dataset. (2) Secondly, we adopted a high-
quality forensic Chain-of-Thought (F-CoT) dataset to
fine tune the model’s instructions, with the goal of
equipping the model with critical thinking ability. The
model is able to cross validateExternal Retrieval Ev-
idencewith its ownInternal Visual Analysis. (3) Fi-
nally, this complex evidence balancing and reasoning
ability is further refined through reinforcement learn-
ing [26]. Our main contributions are threefold.• We proposed high-quality annotation method with
retrieval strategy and construct a detailed Forensic
Knowledge Database (FKD) and a innovative criti-
cal thinking instruction datasets (F-CoT).
• By combining RAG and RL techniques, we de-
signed a MLLM-based critical thinking framework
namedVRAG-DFD. We have explored an effective
training strategy to cultivate critical reasoning skills:
Alignment→SFT→GRPO.
• The proposed framework achieves SOTA & compet-
itive generalization performance on multiple deep-
fake detection benchmarks with excellent interpre-
tation fidelity.
2. Related work
2.1. Traditional Deepfake Detection
Traditional Deepfake detectors typically adopt CNNs
and ViTs as basic frameworks, and employ vari-
ous data processing and transformation methods for
forgery detection, such as frequency transformation
[22, 27, 30], blending operations [11, 24, 41], recon-
struction methods [6, 59], and learning real distribu-
tion with anomaly detection [19, 21]. Although these
strategies have achieved some performance improve-
ments, there are still serious generalization issues. Re-
cently, some methods achieved high generalization
by retaining high-quality visual-language pre-trained
knowledge. Effort [50] employes singular value de-
composition to decompose the original CLIP features
into semantic and artifacts space, thus preserving the
pre-trained knowledge while learning fake patterns.
Despite their intricate structural design, the essence of
these detectors is still a ”black box” for direct binary
classification, and the output results lack interpretabil-
ity. Therefore, combining dedicated Deepfake detec-
tors and MLLM is a very intuitive way to improve the
interpretability.
2.2. MLLMs for Deepfake Detection
In the Deepfake detection task, the introduction
of multimodal large models (MLLMs) provides the
possibility for determining generalization and inter-
pretability [7, 20, 34]. These models are capable of
analyzing and inferring image content through lan-
guage, thereby identifying forgery content. Although
MLLM excels in semantic understanding, it requires
professional low-level information guidance in such
heavy trace DFD detection tasks. To achieve this goal,
VLF-FFD [34],χ2-DFD [7], TruthLens [20] adopted
small deepfake models for information complemen-
tarity. They compensate the main MLLM by supple-
menting with external detectors. But this approach
is redundant in architecture, and knowledge remains
static after training. On the other hand, KFD [52]
and M2F2-Det [13] inject a general forgery knowl-
edge into LLM through prompt learning methods, but
the information content injected by their knowledge is

also isolated and low-quality. Unlike these methods,
this research abandons the static generic forgery trace
list and instead uses the RAG strategy for dynamic
retrieval and annotation. This method provides more
flexible, precise, and targeted external knowledge.
2.3. Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) is a piv-
otal technology designed to facilitate external knowl-
edge expansion for existing Large Language Mod-
els (LLMs) [14]. By leveraging relevant information
retrieved from external knowledge repositories (e.g.,
academic databases, structured knowledge graphs),
RAG can significantly mitigate the occurrence of
hallucinations during the model’s generation process
[18, 38]. This capability enables LLMs to produce
more accurate, factually consistent, and reliable re-
sponses. At present, RAG has been widely applied in
fields such as code generation [57], Visual Question
Answering (VQA) [8, 35], and video understanding
[3, 45]. This paper firstly introduce RAG technology
into Deepfake detection tasks. Experimental results
confirm that the integration of RAG significantly re-
duces the hallucination during the annotation phase,
ultimately contributing to the construction of a more
accurate and trustworthy DFD detector.
3. Method
3.1. Overview
Our VRAG-DFD approach consists of three key com-
ponents. First, we build two datasets: a Forensic
Knowledge Dataset (FKD) with fine-grained artifact
annotations and a Critical Reasoning Dataset with
structured F-CoT annotations. Second, the framework
integrates a Dynamic Forensic Retriever and a Crit-
ical Reasoning MLLM to retrieve relevant evidence
and perform multi-step forensic reasoning. Finally, a
three-stage training pipeline foundational visual align-
ment training, F-CoT supervised fine-tuning, and RL-
GRPO autonomous reasoning reinforcement progres-
sively equips the model with forensic perception, crit-
ical reasoning, and autonomous decision-making ca-
pabilities.
3.2. Datasets
The foundation of our method lies in two hierarchi-
cally complementary datasets: an external knowledge
base for RAG retrieval, and an instruction dataset for
training critical reasoning. Two annotation phases are
demonstrate in Figure 2.
3.2.1. Forensic Knowledge Database
The forensic reasoning capability ofVRAG-DFD
heavily depends on the quality of evidence stored in its
external knowledge base. To support reliable retrieval,
we construct a high-quality, fine-grainedForensic
Knowledge Database (FKD)with detailed expert-
level annotations.We use the FaceForensics++ (FF++) [40]
dataset as the source data for building FKD. The
database contains approximately 18k entries, with
a nearly balanced ratio of forged and real sam-
ples (1:1). Each entry is formatted as[forged
region]: "description of forgery
artifacts...", enabling structured knowledge
retrieval and reasoning.
To ensure annotation precision and forensic relia-
bility, we employ aguide-driven contrastive anno-
tationstrategy:
•Forgery Guides:Based on a comprehensive lit-
erature review, we design detailedForgery Guides
for the four manipulation techniques in FF++ [40]
(DeepFakes [1], Face2Face [46], FaceSwap [2], and
NeuralTextures [47]). Each guide defines the char-
acteristic manipulation patterns of its corresponding
method and describes its typical visual artifact, such
as boundary blurring, texture inconsistency, and il-
lumination anomalies using precise forensic termi-
nology.
•Contrastive Annotation:We employGemini 2.5
Proas the annotation assistant. During annotation,
both forged and pristine frames are presented simul-
taneously, allowing direct comparison. This con-
trastive setup facilitates accurate localization of ma-
nipulated regions and ensures that artifact descrip-
tions strictly follow theForgery Guides.
3.2.2. Critical Reasoning Dataset
To enable the MLLM tocriticallyutilize RAG-
retrieved evidence, we construct an instruction-tuning
dataset specifically designed to teachforensic critical
reasoning.
Data Source.We sample 9,000 image instances from
FF++ [40] as the visual inputs. For each image, we ad-
ditionally retrievektextual evidence entries from the
Forensic Knowledge Database (FKD), which together
form the input pairs used for this fine-tuning stage.
Forensic Chain-of-Thought (F-CoT) Generation.
For each sample, we first obtain aninitial judgment
from the Stage-1 trained model and simultaneously
retrieve the top-kexternal evidence entries from the
Forensic Knowledge Database (FKD). Based on the
correctness of the model’s initial judgment and the va-
lidity of the retrieved RAG evidence, we categorize all
samples into three types. Each type is then annotated
with aForensic Chain-of-Thought (F-CoT)gener-
ated byGemini 2.5 Pro, serving as the gold-standard
reasoning supervision:
•Type 1 (Concurrence / Cross-Verification):[Cor-
rect initial judgment, mixed RAG evidence]. F-CoT
annotations in this category teach the model how to
perform cross-verification among mixed-quality ev-
idence. The model learns to identify and adopt ev-
idence that supports its correct reasoning while re-
jecting misleading counter-evidence, ultimately re-
inforcing its initial conclusion.

———————————————————
❶<Preliminary Visual Analysis>:
Describe independent examination of visual cue.</Preliminary Visual Analysis>———————————————————❷<RAG Reference Information 
Analysis>:Summarizes supportive and opposing evidence retrieved from FKD.</RAG Reference Information Analysis>:———————————————————❸<Fusion, Reasoning, and Decision>:
Integrates both internal observations and external evidence to reach the final forensic conclusion.</Fusion, Reasoning, and Decision>———————————————————❹<Answer>:
The final judgement of the image. 
</Answer>———————————————————Type -2:Evidence -Guided CorrectionType -1:Selective Verification
Type -3: Resilient Rejection✓ Correct Judge
✗Mixed RAG 
➵Learn to verify mixed evidence
✗ Wrong Judge 
✓ Valid RAG
➵Learn to revise with evidence
✗ Wrong Judge 
✗ Misleading RAG
➵Force the model to revert to 
higher -confidence internal know -
ledge and avoid compounding 
RAG errors.
Critical Reasoning Annotation
(F-CoT Generation via Gemini 2.5 Pro)
Top- k
Textual EvidenceTraining Stage -1 
Model
𝑆𝑆𝐼𝐼
𝑇𝑇𝑘𝑘1𝐾𝐾Initial
Judge
Score
Query  Image• Forgery Type: 
└─ NeuralTextures
• Definition:  
└─ NeuralTextures is a facial reenactment technology …• 
Potential Forgery Artifacts:
├─ Core Artifact: Global High -Frequency Detail Loss:
│    ├─ [Cause]: When the GAN/U -Net reconstructs …
│    └─ [Manifestation]:
│            ├─ `Entire face`: Airbrushed feel on the entire face…│            └─ `Loss of skin texture/wrinkles`: Clear ... └─ Local Artifact: Mouth Detail Blurring:
├─ [Cause]: The entire face is re- rendered...     
└─ [Manifestation]:
├─ `Loss of lip texture`: The surface of the ...├─ `Blurry lip borders`: The border line between … └─  `Stiff Lip Shape`: The natural curvature of...
Manipulated Regions : Facial Skin, Lips
Forgery Artifacts :
├─[Facial Skin]: The entire face's skin looks abnormal smooth│         with an airbrushed feel and lacks real pores and texture.└─[Lips]: The lip surface is abnormally smooth and lacks lip 
texture.Potential Forgery Artifacts Reference Guide
FKD
DatabaseF-CoT
Datasets
···
Real-Fake
Pair(a) Annotation phase -1: Forensic Knowledge Data (FKD). (b) Annotation phase -2: Forensic Chain -of-Thought (F -CoT) generation through enumerating contrary.
Forensic Chain -of-Thought (F -CoT)
Gemini 2.5
Figure 2. The anotation phases of the Forensic Knowledge Database (FKD) and Forensic Chain-of-Thought (F-CoT) Dataset.
(Faces are selected from FF++ [40] dataset.)
•Type 2 (RAG-Corrective / Evidence-Guided Cor-
rection):[Incorrect initial judgment, valid RAG ev-
idence]. These samples train the model to recog-
nize when its initial judgment is incorrect and to
leverage valid RAG evidence for correction. The
model is guided to analyze specific artifact-level
cues provided by RAG (e.g.,[mouth region]:
abnormal blending) to verify them against
the image and revise its final decision accordingly.
•Type 3 (Adversarial / Resilient Rejection):[In-
correct initial judgment, misleading RAG evi-
dence]. This type provides the most challenging and
adversarial cases. The F-CoT annotations guide the
model to exercise critical reasoning: first identify-
ing that the retrieved evidence is invalid or mislead-
ing (e.g., the retrieved ”similar” samples are only se-
mantically similar), thenoverridingsuch evidence
and reverting to high-confidence internal forensic
priors (e.g., “the philtrum structure is intact and bio-
logically consistent, hence the face is authentic”) to
arrive at a correct final judgment.
F-CoT Response Structure.All F-CoT annotations
adhere to a unified and structured response format that
explicitly separates visual observation, evidence anal-
ysis, and final decision-making. This structure guides
the model to produce interpretable and logically con-
sistent forensic reasoning:
•Preliminary Visual Analysisdescribes the model’s
independent examination of visual cues (e.g., tex-
ture consistency, artifacts, anomalies).
•RAG Reference Information Analysissumma-
rizes supportive and opposing evidence retrieved
from FKD, along with critical verification of their
reliability.
•Fusion, Reasoning, and Decisionintegrates both
internal observations and external evidence to reach
the final forensic conclusion.
• TheGround Truth Labeldenotes the verified au-thenticity of the sample (e.g., “Real” or “Fake”).
3.3. VRAG-DFD Framework Architecture
Our reasoning framework consists of two core com-
ponents: aDynamic Forensic Retrieverand aCrit-
ical Reasoning MLLM. The former retrieves rele-
vant forensic evidence from an external knowledge
base, while the latter performs high-level reasoning
and decision-making based on both the visual input
and retrieved evidence.
3.3.1. Dynamic Forensic Retriever
The goal of this retriever is to dynamically retrieve
cases from an external knowledge base Forensic
Knowledge Database (FKD) that are highly similar to
the query image. Considering that general pre-trained
models (e.g., CLIP-ViT [37], ResNet [15]) cannot ef-
fectively distinguish deepfake images, we adopted tra-
ditional DFD detector [50] as the embedding model to
achieve this retrieval function. The detailed retrieval
process is shown in Figure 3.
3.3.2. Critical Reasoning MLLM
We employ a pretrained multimodal large language
model (MLLM) as the core reasoning module. Dur-
ing inference, the model takes as input a query im-
age and prompts. More importantly, it also receives
the Top-kexternal evidence retrieved by the Forensic
Retriever. Notably, the retrieved evidence is not the
similar images themselves but the high-qualityF-CoT
textual annotationsassociated with them.
We deliberately choose to use textual evidence
rather than raw image inputs for two reasons: (1)
it prevents redundant visual noise and reduces com-
putational overhead during multimodal reasoning,
and (2) F-CoT annotations explicitly encode expert-
level forensic interpretations (e.g., manipulation re-
gions and their corresponding manipulation artifacts.),

Multimodal Large Language ModelCritical Reasoning
FKD
Database1024- d Feature
Top-kRetrieval Evidence  Top-kRetrieval TokensQuery Image Prompt
 You are a world -class Deepfake Detection expert. 
Your task is to perform **"Cross -Validation"** 
based on "visual evidence" and a "retrieval report" 
to determine whether an image is `Real` or `Fake`.……
Fake -[Chin]: 
Abnormal color and 
inconsistency with 
the face ……Fake -[Eyebrow]: 
Drastic change in 
connection with 
the forehead……Real -[All Face]: 
Normal texture with 
natural lighting.
Fake -[Eyes]: The 
texture of the eyes 
is very blurry……Real -[All Face]: 
Normal texture with natural 
red lighting.1 2 3
4 5
Cosine Similarity❶<Preliminary Visual Analysis>:
Upon observing the overall facial structure and facial feature proportions in the 
Query Image, they appear to be well -proportioned... leading me to initially 
consider this a Real image. … </Preliminary Visual Analysis>
❷<RAG Reference Information Analysis>:
•Opposing Evidence: The highest confidence evidence explicitly points out 
several forged features: ("Fake: Forged areas: eyebrows , eyes, chin", 0.9025). The 
description mentions "a residual image above the eyebrows and "the color of the 
chin is noticeably different from the rest of the skin tone."
•Analysis: The highest -scoring reference provides very specific locations for the 
artifacts ( eyebrows , chin) … </RAG Reference Information Analysis>
❸<Fusion, Reasoning, and Decision>:
 My initial visual analysis leans toward Real. I need to cross -verify the evidence …
•Re-examine: …(residual image above the eyebrows , inconsistent chin color), I 
re-examined the eyebrow and chin areas of the Query_Image.
•Verify: After a more careful inspection, I can now confirm that there is indeed 
an unnatural blur and shadow above the eyebrows . Additionally, the skin texture 
and color in the chin area …
Conclusion: My initial visual intuition was incorrect. The RAG's evidence has been validated … </Fusion, Reasoning, and Decision>
❹<Answer> Fake </Answer>Embedding 
ModelFigure 3. Framework of the proposed VRAG-DFD. Firstly, the dynamic forensic retriever captures top-ksamples from the
multimodal FKD database. Afterwards, MLLM conducts a critical reasoning process (e.g.,①-②-③) based on the text informa-
tion retrieved by RAG , and ultimately outputs the final conclusion (e.g.,④). (Faces are selected from FF++ [40] dataset.)
which are more informative and interpretable for rea-
soning than pixel-level similarity alone. The key idea
of our approach is to leverage image similarity re-
trieval to dynamically obtain textual evidence that is
visually relevant to the query case. The MLLM then
critically compares and integrates these retrieved F-
CoT evidences with the visual content of the query
image to perform forensic reasoning and make the fi-
nal authenticity judgment.
3.4. Three-Stage Training Pipeline
To progressively unlock the model’s forensic reason-
ing capabilities, we design the three-stage training
pipeline illustrated in Figure 3.
3.4.1. Stage-1: Foundational Visual Alignment
Goal.This stage aims to establish the model’s foun-
dational visual perception abilities. Unlike Stage-
2, which relies on fine-grained annotations, Stage-1
leverages large-scale, relatively simple labeled data
for coarse-tuning, thereby laying a solid visual foun-
dation for subsequent complex forensic reasoning
tasks.
Data.We use all frames from 2,500 videos in
FF++, totaling approximately 80k images. No RAG
knowledge base (Sec. 3.2.1) or F-CoT annotations
(Sec. 3.2.2) are employed at this stage.
Task.The model is fine-tuned on a standard Visual
Question Answering (VQA) task with only the ba-
sic classification labels. Experiments demonstrate that
large-scale VQA pre-training significantly enhances
the model’s visual representation capability, which is
essential for Stage-2 when aligning visual features
with fine-grained forensic artifacts.
3.4.2. Stage-2: Critical Reasoning SFT
Goal.Through imitation learning, this stage teaches
the model to execute the three complexForensicChain-of-Thought (F-CoT)reasoning patterns defined
in Sec. 3.2.2.
Data.We use theCritical Reasoning Datasetcon-
structed in Sec. 3.2.2.
Task.This stage forms the core of training the
model’s critical reasoning skills. Building on the vi-
sual capabilities from Stage-1, the model is supervised
with gold-standard F-CoT annotations. The model is
trained to emulate a structured forensic reasoning pro-
cedure. Specifically, the output is formatted as:
1.<Preliminary Visual Analysis>: Inde-
pendently analyze the query image based on Stage-
1 visual capabilities and provide an initial judg-
ment.
2.<RAG Reference Information
Analysis>: Analyze the Top-ktextual ev-
idence retrieved from the Forensic Retriever
(Sec. 3.3.1), clearly distinguishing between
supportiveandopposingevidence.
3.<Fusion, Reasoning, and Decision>:
Perform the core critical reasoning step: cross-
verify the validity of the RAG evidence against the
image, weigh the initial judgment and external ev-
idence, and produce an integrated final reasoning
process.
4.<Answer>: Output the final label: Real or Fake.
By imitating the three data types defined in
Sec. 3.2.2, the model learns to handle conflicting evi-
dence and to master the “forensic-verification” F-CoT
reasoning structure.
3.4.3. Stage-3: Autonomous Reasoning Reinforce-
ment (RL-GRPO)
Goal.The goal of Stage-3 is to elevate the imitation
capabilities learned in Stage-2 to trueAutonomous
Decision-MakingandCritical Thinking. Specifi-
cally, we aim to encourage the model to actively ex-
plore reasoning strategies when encountering conflict-
ing evidence, rather than merely reproducing the SFT

Stage -1: Foundational visual alignment training. Stage -2: Critical rceasoning Supervised Fine -Tuning (SFT). 
F-CoT 
Datasets
(9k)Forensic Chain -of-Thought (F -CoT)
———————————————————
❶<Preliminary Visual Analysis>
———————————————————
❷<RAG Reference Information 
Analysis>
———————————————————
❸<Fusion, Reasoning, and Decision>
———————————————————
❹<Answer>
———————————————————SFT 
TrainingAlignment 
TrainingFF++ Dataset (80k)
Real
Deepfakes FaceSwap
NeuralTextures Face2FaceClassification
<Preliminary Visual Analysis>: 
└─S1_pred
<RAG Reference Information Analysis>: 
└─ RAG Valid
<Answer>:  
└─ Final Pred
G4
<Preliminary Visual Analysis>: 
└─S1_pred
<RAG Reference Information Analysis>: 
└─ RAG Valid
<Answer>:  
└─ Final Pred
G3
<Preliminary Visual Analysis>: 
└─S1_pred
<RAG Reference Information Analysis>: 
└─ RAG Valid
<Answer>:  
└─ Final Pred
G2
<Preliminary Visual Analysis>: 
└─S1_pred
<RAG Reference Information Analysis>: 
└─ RAG Valid
<Answer>:  
└─ Final PredG1𝑅𝑖conflict=+2.0,if𝐴=1∧𝐶=1,
+1.0,if𝐴=1∧𝐶=0,
−1.0,if𝐴=0∧𝐶=0,
−2.0,if𝐴=0∧𝐶=1.
𝑅𝑖format=ቊ1.0, if𝑓𝑜𝑙𝑙𝑜𝑤
0.0,if𝑁𝑂𝑇𝑓𝑜𝑙𝑙𝑜𝑤𝛼
𝛽𝑅𝑖allStage -3: Autonomous reasoning reinforcement learning (RL -GRPO) with 9k annotated data. 
Sampling
GRPO 
Training
Format RewardConflict Reward1
𝐺෍
𝑖=1𝐺
𝑅𝑖all
Figure 4. Training stages of VRAG-DFD. Stage-1 aims to equip the MLLM with basic forgery artifacts recognition capabilities.
Stage-2 aims to guide the model to predict in the form of F-CoT. Stage-3 adopts GRPO with detailed reward function designs
to encourage the MLLM to have the capabilities to critical reasoning given top-kRAG evidence. (Faces are selected from
FF++ [40] dataset.)
demonstration paths.
Data.To mitigate potential overfitting from SFT (e.g.,
Advantage≈0), Stage-3 is conducted on data un-
seen during Stage-1 and Stage-2. This ensures suf-
ficient optimization space for exploring new data dis-
tributions. We use previously unobserved frames from
FF++ [40] as the training set.
Process-Aware Reward Function.The Process-
Aware Reward Function as the sum of two compo-
nents: Format Reward and Conflict Reward. For-
mat Reward mandates the model to follow theF-CoT
structure. Conflict Reward dynamically evaluates the
reasoning paths generated by the policy model relying
on two key signals:
1. The model’s preliminary visual analysis, which
reveals its inherent preference and initial prediction
(s1pred).
2. The validity of the retrieved RAG evidence (e.g.,
assessed via majority voting or annotation consis-
tency), which reflects the reliability of external knowl-
edge.
By comparings1 predwith the RAG evidence, we
can detect conflicts: if the model correctly resolves
conflicts, it demonstrates successful integration of in-
ternal visual knowledge and external evidence, ex-
hibiting true critical reasoning; if the model fails, pro-
ducing an incorrect prediction, it indicates potential
over-reliance or over-rejection of RAG evidence.
Specifically, we defineA∈ {0,1}as an indicator
of whether the model’s final answer is correct (A= 1
denotes correctness). We further defineC∈ {0,1}to
represent whether there exists a conflict between the
model’s preliminary visual judgment (s1 pred) and
the evidence retrieved via RAG (rag correct), whereC= 1indicates a conflict. Our conflict reward func-
tionR conflict is formulated as follows:
Rconflict =

+2.0,ifA= 1∧C= 1
+1.0,ifA= 1∧C= 0
−1.0,ifA= 0∧C= 0
−2.0,ifA= 0∧C= 1(1)
This incentive structure creates a clear reinforce-
ment loop:
•Highest reward (+2.0):Assigned tocritical cor-
rectness, where the model correctly resolves a con-
flict (C= 1) between internal perception and exter-
nal evidence (A= 1).
•Lowest reward (-2.0):Strongly penalizescritical
errors, where the model fails under conflicting evi-
dence (C= 1,A= 0).
Through this design, the model is explicitly en-
couraged during the GRPO stage to explore and re-
inforce high-value critical reasoning behaviors, rather
than merely repeating thesimple correctnesspatterns
learned during the SFT stage.
For each samplei, we define the final reward as the
sum of a conflict reward and a format reward:
Ri=α r i+β f i.(2)
withαandβdenoting the weighting coefficients for
the respective reward components.
For a batch ofGsamples, we take the mean reward
as the batch-level supervision signal:
R=1
GGX
i=1(α ri+β f i).(3)

Table 1. Cross-dataset performance comparison with SOTA face forgery detection methods on CDF-v1 [25], CDF-v2 [25],
DFDC [9], FFIW [58], and WDF [60]. All methods are trained on FF++ [40]. The best results are shown inbold, and the
second results are underlined . The Video-level AUC metric is adopted for comparison.
Method Venue Explainable Research Topic CDF-v1 CDF-v2 DFDC FFIW WDF
F3-Net [36] ECCV 2020 ✗ Frequency + CNN 81.11 77.92 67.35 70.11 72.80
LTW [43] AAAI 2021 ✗ Meta Training - 77.14 69.00 76.63 -
SPSL [28] CVPR 2021 ✗ Phase Spectrum - 79.90 77.00 79.40 70.20
SRM [29] CVPR 2021 ✗ Attention + FPN - 84.00 69.50 80.60 72.20
PCL+I2G [54] ICCV 2021 ✗ Blending - 90.03 67.52 - -
SBI [41] CVPR 2022 ✗ Blending 93.44 93.18 72.42 84.83 -
CORE [33] CVPR 2022 ✗ Augmentation - 80.90 72.10 71.00 72.40
DCL [44] AAAI 2022 ✗ Contrastive Learning - 82.30 76.71 71.14 71.14
SeeABLE [21] ICCV 2023 ✗ Contrastive Learning - 87.30 75.90 - -
F2Trans [31] TIFS 2023 ✗ Attention + Wavelet 86.29 89.87 - - -
AltFreezing [48] CVPR 2023 ✗ Spatial + Temporal 88.48 89.50 64.75 - -
AUNet [5] CVPR 2023 ✗ Facial AU - 92.77 73.82 81.45 -
UCF [49] ICCV 2023 ✗ Disentanglement 86.08 83.73 75.11 69.70 77.42
LAA-Net [32] CVPR 2024 ✗ Attention + FPN - 95.40 - - -
FreqBlender [56] NeurIPS 2024 ✗ Frequency + Blending - 94.59 74.59 86.14 -
UDD [10] AAAI 2025 ✗ Token Shuffle + Mixing - 93.10 81.20 - -
LESB [42] WACVW 2025 ✗ Blending - 93.13 71.98 83.01 -
Effort [50] ICML 2025 ✗ Fine-tune CLIP-ViT 96.05 95.60 84.30 92.10 84.80
χ2-DFD [7] NeurIPS 2025 ✓ MLLM + Detector - 95.5085.3086.70 86.40
KFD [52] ICML 2025 ✓ MLLM + Static Knowledge 97.62 94.71 79.12 - -
M2F2-Det [13] CVPR 2025 ✓ MLLM + Static Knowledge - 95.10 - 88.70 -
OursCVPRF 2026 ✓ MLLM +Dynamic RAG 99.60 95.9781.8293.49 88.96
This design rewards successful conflict resolution
and critical reasoning.
4. Experiment
4.1. Basic Settings
Datasets.All training stages are conducted on Face-
Forensics++ (FF++) dataset [40]. Following the split
method with [51], we keep the same data partition-
ing settings in all three training stages. Specifi-
cally, Stage-1 (Alignment FT) utilize 2,500 videos
from the FF++ training set. Stage-2 (Critical SFT)
and Stage-3 (RL-GRPO) are trained on the remain-
ing 1,000 videos. To rigorously evaluate the general-
ization capability of the proposed method, we adopt
a strict cross-dataset evaluation setting: no images or
videos from any of the evaluation datasets are used
during any SFT stage. We report performance on
five widely adopted Deepfake detection benchmarks:
Celeb-DF-v1 (CDF-v1) [25], Celeb-DF-v2 (CDF-v2)
[25], FFIW [58], DFDC [9] and WildDeepfake (WDF)
[60]. These datasets cover diverse manipulation types,
acquisition conditions, and levels of visual realism,
enabling a comprehensive assessment of robustness
and transferability.
Baselines. As shown in Table 1, we compare VRAG-
DFD with two categories of state-of-the-art methods.
Traditional SOTA detectors contain F3-Net [36], LTW
[43], SPSL [28], SRM [29], PCL+I2G [54], DCL [44],
SBI [41], CORE [33], F2Trans [31], AUNet [5], Alt-
Freezing [48], SeeABLE [21], UCF [49], LAA-Net
[32], FreqBlender [56], UDD [10], LESB [42] and
Effort [50]. MLLM-based Detectors containχ2-DFD[7], KFD [52] and M2F2-DET [13].
Evaluation Metrics.We use the Area Under the Re-
ceiver Operating Characteristic Curve (AUC) as the
primary evaluation metric for classification and gen-
eralization performance. Following [13], we adopted
GPT-4o as the automatic judge to assess the quality of
generated explanations. Detailed qualitative compar-
isons are provided in the supplementary materials.
Implementation Details.The proposed VRAG-DFD
framework is based on the Qwen2.5-VL MLLM [4].
Low-Rank Adaptation (LoRA) [17] is adopted for all
training stages withr= 128, α= 256. Training con-
figurations of three stages are summarized in Table 2.
Table 2. Training configurations of three training stages.
Stage Objective Epoch lr Batch Param.
Stage-1 Alignment 35×10−5512 -
Stage-2 SFT 23×10−564 -
Stage-3 GRPO 11×10−632β= 0.001
4.2. Performance Analysis
Main Results.Cross-dataset generalization perfor-
mance is summarized in Table 1. Our VRAG-DFD
framework achieves state-of-the-art results on four
out of five evaluation datasets (CDF-v1 [25], CDF-
v2 [25], FFIW [58], and WDF [60]). These results
strongly demonstrate that our approach leveraging
RAG-based dynamic evidence retrieval coupled with
critical reasoning significantly enhances the model’s
ability to generalize to unseen forgery types and com-
plex data distributions.
Analysis on the DFDC Dataset.We observe that
the performance on the DFDC dataset does not reach
SOTA. This limitation is expected under ourrestricted

Table 3. Ablation study of RAG (AUC %).w/oRAG re-
moves the forensic retriever and all associated critical rea-
soning components.
MetricsTest Set AUC
CDF1 CDF2 DFDC FFIW WDF
w/oRAG Module 91.13 87.85 72.42 78.62 71.62
VRAG-DFD 99.60 95.97 81.82 93.49 88.96
Table 4. Ablation study of the three-stage training pipeline.
We report AUC (%) on five generalization benchmarks.
MetricsTest Set AUC
CDF1 CDF2 DFDC FFIW WDF
Stage-1 96.77 94.03 78.54 90.89 87.63
Stage-1&2 91.5395.8 81.58 92.37 89.13
VRAG-DFD 99.60 95.97 81.82 93.49 88.96
experimental setting, where the RAG database is
exclusively built from FF++ samples. The DFDC
dataset contains a large number of samples that dif-
fer from FF++ in both visual domain characteristics
and forgery artifacts. Consequently, our forensic ev-
idence retriever fails to retrieve highly relevant his-
torical cases. The external evidence accessible to the
MLLM is either of low quality or irrelevant, which in-
terferes with its critical reasoning process and leads
to degraded performance. This observation, in fact,
validates our central claim: the detection capability of
MLLMs is strongly dependent on the quality of the
external evidence they can access.
4.3. Ablation Studies
We conduct a series of comprehensive ablation studies
to deconstruct the VRAG-DFD framework and vali-
date the effectiveness of each design choice.
Effectiveness of Forensic RAG.To quantify the con-
tribution of our core RAG-based forensic retrieval
component, we evaluate a ”w/oRAG” variant that re-
lies solely on the MLLM’s internal knowledge for de-
tection. As shown in Table 3, removing the RAG
leads to a substantial drop in performance across all
datasets, demonstrating that dynamic and targeted ex-
ternal evidence is crucial for robust forensic detection.
Effectiveness of Three-Stage Pipeline.Table 4 re-
ports the ablation results across the three training
stages. Using only Stage-1, the model acquires basic
visual perception but lacks reasoning ability, leading
to limited performance. Incorporating Stage-2 (Crit-
ical SFT) yields a substantial improvement, demon-
strating the effectiveness of the F-CoT dataset and
structured reasoning format (Sec. 3.2.2) in enabling
forensic analysis. Further adding Stage-3 (RL-GRPO)
enhances critical thinking, indicating that the process-
aware reward model (Sec. 3.4.3) promotes a transition
from imitation to autonomous decision-making. Over-
all, the consistent performance gains confirm that each
stage is necessary and complementary.
Comparison of Forensic Retrievers.To ensure thatTable 5. Ablation study of retriever (AUC %). ‘CLIP-LoRA’
replaces Effort [50] with a standard CLIP model fine-tuned
by LoRA.
RetrieversTest Set AUCAvg.CDF1 CDF2 DFDC FFIW WDF
Effort [50] 99.6095.9781.82 93.4988.96 91.62
CLIP-LoRA[17] 100.0095.7782.47 94.0088.42 92.13
Table 6. Explanation quality evaluation. GPT-4o and Gem-
ini 2.5 Pro serve as evaluators. VRAG-DFD achieves the
highest average score.
Model GPT-4o Gemini2.5 Pro Avg.
GPT-4o 4.60 3.25 3.93
Gemini2.5 Pro 7.31 7.02 7.16
VRAG-DFD 7.55 7.78 7.66
the superiority of VRAG-DFD does not rely on a par-
ticular state-of-the-art detector (i.e., Effort [50]), We
also compare the impact of different retrievers on the
final performance. Specifically, we evaluate an Effort-
based retriever [50], which employs orthogonal sub-
space decomposition, against a standard CLIP-ViT
[37] retriever fine-tuned with LoRA [17] on the same
dataset. As shown in Table 5, the clip-lora-based
retriever achieves superior overall performance. It
demonstrates that VRAG-DFD does not depend on Ef-
fort’s specialized SVD design; instead, a simpler and
more generic tuning method (LoRA) unlocks greater
potential for forensic retrieval.
Evaluation of Explanation QualityTo quantitatively
evaluate the reliability of model-generated explana-
tions, we conduct a multi-dimensional assessment on
100 randomly selected samples. Explanations are pro-
duced by three models: GPT-4o, Gemini 2.5 Pro, and
our VRAG-DFD framework. We further employ GPT-
4o and Gemini 2.5 Pro as independent evaluators to
score each explanation along three standard dimen-
sions (0–3 points each): accuracy, faithfulness, and
professionalism. These criteria collectively measure
whether the prediction is correct, whether the reason-
ing is grounded in visual evidence, and whether the
explanation follows forensic conventions. The results
in Table 6 show that VRAG-DFD substantially outper-
forms general-purpose MLLMs.
5. Conclusion
To achieve the high-quality associated forgery knowl-
edge and critical reasoning abilities in MLLM-based
DFD detection, we firstly constructed two datasets:
Forensic Knowledge Database (FKD) and Forensic
Chain-of-Thought Dataset (F-CoT). For critical CoT
construction, we propose the VRAG-DFD with accu-
rate dynamic forgery knowledge retrieval and pow-
erful critical reasoning capabilities. VRAG-DFD
achieved SOTA and competitive performance on DFD
generalization benchmarks. This research confirms
the application prospects of RAG in the field of DFD.

References
[1] Deepfakes.https : / / github . com /
deepfakes / faceswap, 2020. Accessed:
2020-05-10. 3
[2] Faceswap.https : / / github . com /
MarekKowalski/FaceSwap, 2020. Accessed:
2020-05-10. 3
[3] Md Adnan Arefeen, Biplob Debnath, Md Yusuf Sar-
war Uddin, and Srimat Chakradhar. irag: Advancing
rag for videos with an incremental approach. InPro-
ceedings of the 33rd ACM International Conference
on Information and Knowledge Management, pages
4341–4348, 2024. 3
[4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang,
Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie
Wang, Jun Tang, et al. Qwen2. 5-vl technical report.
arXiv preprint arXiv:2502.13923, 2025. 2, 7
[5] Weiming Bai, Yufan Liu, Zhipeng Zhang, Bing Li, and
Weiming Hu. Aunet: Learning relations between ac-
tion units for face forgery detection. InProceedings
of the IEEE/CVF conference on computer vision and
pattern recognition, pages 24709–24719, 2023. 7
[6] Junyi Cao, Chao Ma, Taiping Yao, Shen Chen,
Shouhong Ding, and Xiaokang Yang. End-to-end
reconstruction-classification learning for face forgery
detection. InProceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages
4113–4122, 2022. 1, 2
[7] Yize Chen, Zhiyuan Yan, Guangliang Cheng, Kangran
Zhao, Siwei Lyu, and Baoyuan Wu. X2-dfd: A frame-
work for explainable and extendable deepfake detec-
tion.arXiv preprint arXiv:2410.06126, 2024. 2, 7
[8] Matthew Dahl, Varun Magesh, Mirac Suzgun, and
Daniel E Ho. Large legal fictions: Profiling legal hal-
lucinations in large language models.Journal of Legal
Analysis, 16(1):64–93, 2024. 3
[9] Brian Dolhansky, Joanna Bitton, Ben Pflaum, Jikuo
Lu, Russ Howes, Menglin Wang, and Cristian Can-
ton Ferrer. The deepfake detection challenge (dfdc)
dataset.arXiv preprint arXiv:2006.07397, 2020. 7
[10] Xinghe Fu, Zhiyuan Yan, Taiping Yao, Shen Chen,
and Xi Li. Exploring unbiased deepfake detection via
token-level shuffling and mixing. InProceedings of
the AAAI Conference on Artificial Intelligence, pages
3040–3048, 2025. 7
[11] Xinghe Fu, Zhiyuan Yan, Taiping Yao, Shen Chen,
and Xi Li. Exploring unbiased deepfake detection via
token-level shuffling and mixing. InProceedings of
the AAAI Conference on Artificial Intelligence, pages
3040–3048, 2025. 1, 2
[12] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza,
Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron
Courville, and Yoshua Bengio. Generative adversar-
ial networks.Communications of the ACM, 63(11):
139–144, 2020. 1
[13] Xiao Guo, Xiufeng Song, Yue Zhang, Xiaohong Liu,
and Xiaoming Liu. Rethinking vision-language model
in face forensics: Multi-modal interpretable forged
face detector. InProceedings of the Computer Vision
and Pattern Recognition Conference, pages 105–116,
2025. 2, 7[14] Hangfeng He, Hongming Zhang, and Dan Roth. Re-
thinking with retrieval: Faithful large language model
inference.ArXiv, abs/2301.00303, 2022. 3
[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian
Sun. Deep residual learning for image recognition. In
Proceedings of the IEEE conference on computer vi-
sion and pattern recognition, pages 770–778, 2016. 4
[16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denois-
ing diffusion probabilistic models.Advances in neural
information processing systems, 33:6840–6851, 2020.
1
[17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, et al. Lora: Low-rank adaptation of
large language models.ICLR, 1(2):3, 2022. 7, 8
[18] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu,
Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. Survey of hallucination in
natural language generation.ACM computing surveys,
55(12):1–38, 2023. 3
[19] Hasam Khalid and Simon S Woo. Oc-fakedect: Clas-
sifying deepfakes using one-class variational autoen-
coder. InProceedings of the IEEE/CVF conference on
computer vision and pattern recognition workshops,
pages 656–657, 2020. 1, 2
[20] Rohit Kundu, Athula Balachandran, and Amit K Roy-
Chowdhury. Truthlens: Explainable deepfake detec-
tion for face manipulated and fully synthetic data.
arXiv preprint arXiv:2503.15867, 2025. 2
[21] Nicolas Larue, Ngoc-Son Vu, Vitomir Struc, Peter
Peer, and Vassilis Christophides. Seeable: Soft dis-
crepancies and bounded contrastive learning for ex-
posing deepfakes. InProceedings of the IEEE/CVF
International Conference on Computer Vision, pages
21011–21021, 2023. 1, 2, 7
[22] Jiaming Li, Hongtao Xie, Jiahong Li, Zhongyuan
Wang, and Yongdong Zhang. Frequency-aware dis-
criminative feature learning supervised by single-
center loss for face forgery detection. InProceedings
of the IEEE/CVF conference on computer vision and
pattern recognition, pages 6458–6467, 2021. 2
[23] Lingzhi Li, Jianmin Bao, Hao Yang, Dong Chen,
and Fang Wen. Faceshifter: Towards high fidelity
and occlusion aware face swapping.arXiv preprint
arXiv:1912.13457, 2019. 1
[24] Lingzhi Li, Jianmin Bao, Ting Zhang, Hao Yang,
Dong Chen, Fang Wen, and Baining Guo. Face x-ray
for more general face forgery detection. InProceed-
ings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 5001–5010, 2020. 1, 2
[25] Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, and
Siwei Lyu. Celeb-df: A large-scale challenging
dataset for deepfake forensics. InProceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, pages 3207–3216, 2020. 7
[26] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang,
Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi
Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-
v3 technical report.arXiv preprint arXiv:2412.19437,
2024. 2
[27] Honggu Liu, Xiaodan Li, Wenbo Zhou, Yuefeng Chen,
Yuan He, Hui Xue, Weiming Zhang, and Nenghai
Yu. Spatial-phase shallow learning: rethinking face

forgery detection in frequency domain. InProceed-
ings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 772–781, 2021. 2
[28] Honggu Liu, Xiaodan Li, Wenbo Zhou, Yuefeng Chen,
Yuan He, Hui Xue, Weiming Zhang, and Nenghai
Yu. Spatial-phase shallow learning: rethinking face
forgery detection in frequency domain. InProceed-
ings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 772–781, 2021. 7
[29] Yuchen Luo, Yong Zhang, Junchi Yan, and Wei
Liu. Generalizing face forgery detection with high-
frequency features. InProceedings of the IEEE/CVF
conference on computer vision and pattern recogni-
tion, pages 16317–16326, 2021. 7
[30] Yuchen Luo, Yong Zhang, Junchi Yan, and Wei
Liu. Generalizing face forgery detection with high-
frequency features. InProceedings of the IEEE/CVF
conference on computer vision and pattern recogni-
tion, pages 16317–16326, 2021. 2
[31] Changtao Miao, Zichang Tan, Qi Chu, Huan Liu,
Honggang Hu, and Nenghai Yu. F2trans: High-
frequency fine-grained transformer for face forgery de-
tection.IEEE Transactions on Information Forensics
and Security, 18:1039–1051, 2023. 7
[32] Dat Nguyen, Nesryne Mejri, Inder Pal Singh, Polina
Kuleshova, Marcella Astrid, Anis Kacem, Enjie Ghor-
bel, and Djamila Aouada. Laa-net: Localized arti-
fact attention network for quality-agnostic and gen-
eralizable deepfake detection. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 17395–17405, 2024. 7
[33] Yunsheng Ni, Depu Meng, Changqian Yu, Chengbin
Quan, Dongchun Ren, and Youjian Zhao. Core: Con-
sistent representation learning for face forgery detec-
tion. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR)
Workshops, pages 12–21, 2022. 7
[34] Siran Peng, Zipei Wang, Li Gao, Xiangyu Zhu,
Tianshuo Zhang, Ajian Liu, Haoyuan Zhang, and
Zhen Lei. Mllm-enhanced face forgery detection:
A vision-language fusion solution.arXiv preprint
arXiv:2505.02013, 2025. 2
[35] Hongxu Pu, Xincong Yang, Jing Li, and Runhao Guo.
Autorepo: A general framework for multimodal llm-
based automated construction reporting.Expert Sys-
tems with Applications, 255:124601, 2024. 3
[36] Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen,
and Jing Shao. Thinking in frequency: Face forgery
detection by mining frequency-aware clues. InEu-
ropean conference on computer vision, pages 86–103.
Springer, 2020. 7
[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sas-
try, Amanda Askell, Pamela Mishkin, Jack Clark, et al.
Learning transferable visual models from natural lan-
guage supervision. InInternational conference on ma-
chine learning, pages 8748–8763. PmLR, 2021. 4, 8
[38] Vikas Raunak, Arul Menezes, and Marcin Junczys-
Dowmunt. The curious case of hallucinations in neural
machine translation.ArXiv, abs/2104.06683, 2021. 3
[39] Robin Rombach, Andreas Blattmann, Dominik
Lorenz, Patrick Esser, and Bj ¨orn Ommer. High-
resolution image synthesis with latent diffusion mod-els. InProceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages
10684–10695, 2022. 1
[40] Andreas Rossler, Davide Cozzolino, Luisa Verdoliva,
Christian Riess, Justus Thies, and Matthias Nießner.
Faceforensics++: Learning to detect manipulated fa-
cial images. InProceedings of the IEEE/CVF inter-
national conference on computer vision, pages 1–11,
2019. 1, 3, 4, 5, 6, 7
[41] Kaede Shiohara and Toshihiko Yamasaki. Detecting
deepfakes with self-blended images. InProceedings
of the IEEE/CVF conference on computer vision and
pattern recognition, pages 18720–18729, 2022. 1, 2, 7
[42] Elahe Soltandoost, Richard Plesh, Stephanie Schuck-
ers, Peter Peer, and Vitomir ˇStruc. Extracting local in-
formation from global representations for interpretable
deepfake detection. InProceedings of the Winter Con-
ference on Applications of Computer Vision, pages
1629–1639, 2025. 7
[43] Ke Sun, Hong Liu, Qixiang Ye, Yue Gao, Jianzhuang
Liu, Ling Shao, and Rongrong Ji. Domain general face
forgery detection by learning to weight. InProceed-
ings of the AAAI conference on artificial intelligence,
pages 2638–2646, 2021. 7
[44] Ke Sun, Taiping Yao, Shen Chen, Shouhong Ding,
Jilin Li, and Rongrong Ji. Dual contrastive learning
for general face forgery detection. InProceedings of
the AAAI conference on artificial intelligence, pages
2316–2324, 2022. 7
[45] Xichen Tan, Yunfan Ye, Yuanjing Luo, Qian Wan,
Fang Liu, and Zhiping Cai. Rag-adapter: A plug-
and-play rag-enhanced framework for long video un-
derstanding.arXiv preprint arXiv:2503.08576, 2025.
3
[46] Justus Thies, Michael Zollhofer, Marc Stamminger,
Christian Theobalt, and Matthias Nießner. Face2face:
Real-time face capture and reenactment of rgb videos.
InProceedings of the IEEE conference on computer vi-
sion and pattern recognition, pages 2387–2395, 2016.
1, 3
[47] Justus Thies, Michael Zollh ¨ofer, and Matthias
Nießner. Deferred neural rendering: Image synthesis
using neural textures.Acm Transactions on Graphics
(TOG), 38(4):1–12, 2019. 1, 3
[48] Zhendong Wang, Jianmin Bao, Wengang Zhou,
Weilun Wang, and Houqiang Li. Altfreezing for more
general video face forgery detection. InProceedings
of the IEEE/CVF conference on computer vision and
pattern recognition, pages 4129–4138, 2023. 7
[49] Zhiyuan Yan, Yong Zhang, Yanbo Fan, and Baoyuan
Wu. Ucf: Uncovering common features for gener-
alizable deepfake detection. InProceedings of the
IEEE/CVF international conference on computer vi-
sion, pages 22412–22423, 2023. 7
[50] Zhiyuan Yan, Jiangming Wang, Peng Jin, Ke-
Yue Zhang, Chengchun Liu, Shen Chen, Taiping
Yao, Shouhong Ding, Baoyuan Wu, and Li Yuan.
Orthogonal subspace decomposition for generaliz-
able ai-generated image detection.arXiv preprint
arXiv:2411.15633, 2024. 1, 2, 4, 7, 8
[51] Zhiyuan Yan, Taiping Yao, Shen Chen, Yandan Zhao,
Xinghe Fu, Junwei Zhu, Donghao Luo, Chengjie
Wang, Shouhong Ding, Yunsheng Wu, et al. Df40:

Toward next-generation deepfake detection.Advances
in Neural Information Processing Systems, 37:29387–
29434, 2024. 7
[52] Peipeng Yu, Jianwei Fei, Hui Gao, Xuan Feng, Zhihua
Xia, and Chip Hong Chang. Unlocking the capabili-
ties of large vision-language models for generalizable
and explainable deepfake detection.arXiv preprint
arXiv:2503.14853, 2025. 2, 7
[53] Yue Zhang, Ben Colman, Xiao Guo, Ali Shahriyari,
and Gaurav Bharaj. Common sense reasoning for
deepfake detection. InEuropean conference on com-
puter vision, pages 399–415. Springer, 2024. 2
[54] Tianchen Zhao, Xiang Xu, Mingze Xu, Hui Ding,
Yuanjun Xiong, and Wei Xia. Learning self-
consistency for deepfake detection. InProceedings of
the IEEE/CVF international conference on computer
vision, pages 15023–15033, 2021. 7
[55] Wenliang Zhao, Yongming Rao, Weikang Shi, Zuyan
Liu, Jie Zhou, and Jiwen Lu. Diffswap: High-fidelity
and controllable face swapping via 3d-aware masked
diffusion. InProceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition,
pages 8568–8577, 2023. 1
[56] Jiaran Zhou, Yuezun Li, Baoyuan Wu, Bin Li, Junyu
Dong, et al. Freqblender: Enhancing deepfake de-
tection by blending frequency knowledge.Advances
in Neural Information Processing Systems, 37:44965–
44988, 2024. 7
[57] Shuyan Zhou, Uri Alon, Frank F Xu, Zhengbao Jiang,
and Graham Neubig. Docprompting: Generating code
by retrieving the docs. InThe Eleventh International
Conference on Learning Representations, 2022. 3
[58] Tianfei Zhou, Wenguan Wang, Zhiyuan Liang, and
Jianbing Shen. Face forensics in the wild. InPro-
ceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), pages 5778–
5788, 2021. 7
[59] Xiangyu Zhu, Hao Wang, Hongyan Fei, Zhen Lei, and
Stan Z Li. Face forgery detection by 3d decomposi-
tion. InProceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 2929–
2939, 2021. 1, 2
[60] Bojia Zi, Minghao Chang, Jingjing Chen, Xingjun Ma,
and Yu-Gang Jiang. Wilddeepfake: A challenging
real-world dataset for deepfake detection. InProceed-
ings of the 28th ACM international conference on mul-
timedia, pages 2382–2390, 2020. 7

VRAG-DFD: Verifiable Retrieval-Augmentation for MLLM-based
Deepfake Detection
Supplementary Material
A. Implementation Details
A.1. Retrieval and Inference Settings
In our dynamic forensic retrieval module, we set the
number of retrieved evidence items tok= 5. This
choice is based on empirical observations on the vali-
dation set, wherek= 5provides an effective trade-off
between supplying sufficient supportive evidence and
maintaining efficient MLLM reasoning. Larger val-
ues ofktend to introduce unnecessary noise due to
excessively long context windows, while smaller val-
ues lead to insufficient contextual grounding.
A.2. Training Configuration and Data Aug-
mentation
As described in the main paper, the training pipeline
of VRAG-DFD consists of three stages. To maximize
model performance, we adopt tailored training strate-
gies for each stage.
Stage-1: Foundational Visual Alignment.We use
approximately 80,000 images from the FaceForen-
sics++ (FF++) training set.
•Training Strategy.This stage aims to establish the
model’s foundational visual sensitivity to forgery
cues. We therefore perform LoRA fine-tuning on
the ViT (vision encoder), the Aligner (projection
module), and the LLM.
•Data Augmentation.To enhance the visual en-
coder’s robustness and mitigate overfitting, we ap-
ply a set of augmentations implemented using the
Albumentations library, such as Horizontal Flip, Im-
age Compression, HueSaturationValue Adjustment.
Stage-2 & Stage-3: Critical Reasoning Training.
Both Stage-2 and Stage-3 are trained on 9,000 sam-
ples sampled from the FF++ dataset. In Stage-2, these
samples are used for supervised fine-tuning (SFT)
with F-CoT annotations.
•Training Strategy.During these stages, we freeze
all ViT parameters and only lora finetune the
Aligner and LLM. This design preserves the general
visual features learned in Stage-1, while encourag-
ing the model to focus on logical reasoning, evi-
dence aggregation, and conflict resolution.
•Data Augmentation.To ensure that the model can
capture extremely subtle pixel-level forgery traces,
no data augmentation is applied in Stage 2 and
Stage 3. This avoids introducing artificial distor-
tions that could obscure genuine forgery artifacts.
B. Database Construction Details
To support the VRAG-DFD framework, we con-
struct two core datasets: (1) theForensic KnowledgeDatabase (FKD)for retrieval, and (2) theForensic
Chain-of-Thought (F-CoT)dataset for critical reason-
ing training.
B.1. Construction of the Forensic Knowledge
Database (FKD)
Data Sources and Balancing Strategy.The origi-
nal FF++ dataset exhibits a Real:Fake ratio of approx-
imately 1:4. To build a category-balanced retrieval
database, we adopt the following sampling strategy,
which produces a balanced 1:1 distribution of real and
manipulated samples:
•Fake Frames:From each video of the four manip-
ulation types (DeepFakes, Face2Face, FaceSwap,
NeuralTextures), we randomly sample around 3
frames, resulting in approximately 8,000 fake
frames.
•Real Frames:From each real video, we sample ap-
proximately 12 frames, also yielding around 8,000
real frames.
Expert-Level Annotation.To ensure high-quality
and professional annotations, we adopt a combina-
tion of contrastive annotation and a structured forgery
guideline. Using Gemini 2.5 Pro, we input each target
image together with its corresponding real/fake coun-
terpart and instruct the model to produce detailed de-
scriptions of forged regions, guided by manipulation-
specific characteristics (e.g., texture discontinuities in
NeuralTextures). The full prompt is shown in the END
of this supplementary material.
B.2. Construction of the Forensic Chain-of-
Thought (F-CoT) Dataset
Data Source.We select 1,000 videos from the FF++
training split that are not used during Stage 1 training.
From each video, 9 frames are sampled, resulting in a
dataset of 9,000 samples.
Reference Information Generation.For each sam-
ple, we retrieve the top-kmost relevant images from
FKD (excluding itself) and use their text annotations
as external reference information.
Critical Logic Construction.To encourage the
model to reason critically about retrieved evidence, we
categorize each sample into one of three types based
on the Stage 1 prediction and the quality of retrieved
evidence. Gemini 2.5 Pro, acting as a “teacher” with
access to ground-truth labels, is prompted to produce
corresponding chain-of-thought reasoning:

Table 7.Data Quality Comparison between FKD and
DDVQA.Replacing FKD with DDVQA leads to a sig-
nificant performance drop, indicating that forensic-grade
knowledge is essential for effective reasoning.
MetricsTest Set AUCAvg.CDF1 CDF2 DFDC FFIW WDF
DD-VQA [53] 96.17 94.83 79.48 92.50 85.88 89.77
FKD (Ours) 99.60 95.97 81.82 93.49 88.96 91.62
•Type-1: Cross-Validation.Condition:Stage 1 pre-
dicts correctly.Objective:Teach the model to verify
its visual judgment using supportive retrieved evi-
dence while ignoring irrelevant noise.
•Type-2: Evidence-Guided Correction.Condition:
Stage 1 predicts incorrectly, but retrieved evidence
contains strong, reliable cues.Objective:Guide the
model to recognize its initial mistake and rely on
high-quality external evidence for correction.
•Type-3: Resilient Rejection.Condition:Stage 1
predicts incorrectly, and retrieved evidence is mis-
leading (e.g., semantically similar but different ma-
nipulation type).Objective:Train the model to
identify unreliable evidence, reject misleading cues,
and fall back to its internal visual knowledge for the
final judgment.
The detailed F-CoT generation prompt in the END
of this supplementary material.
C. Additional Experimental Results
C.1. Data Quality Comparison: FKD vs.
DDVQA.
To further examine how knowledge quality influences
retrieval-based reasoning, we replace the FKD re-
trieval corpus in VRAG-DFD with the textual descrip-
tions from the DDVQA [53] dataset and conduct ex-
periments under identical settings. As shown in Ta-
ble 7, replacing FKD with DDVQA results in a sub-
stantial performance drop. This observation demon-
strates that the fine-grained, expert-level forensic an-
notations contained in FKD are crucial for enhanc-
ing reasoning ability. In contrast, the descriptions in
DDVQA are relatively coarse and lack professional
forensic knowledge, making them insufficient for pro-
viding stable and reliable forgery cues. Overall, these
findings underscore the critical importance of high-
quality forensic knowledge for cross-dataset general-
ization and robust reasoning.
C.2. Contribution of the RAG Module Across
Stages
To quantify the efficiency of external knowledge re-
trieval, we conduct an ablation study by selectively
activating the RAG module at each training stage. The
results in Table 8 demonstrate substantial and con-
sistent gains across all stages. This confirms that
the RAG mechanism is an integral component of ourTable 8.Ablation study on the impact of the RAG mod-
ule across three training stages.We report the AUC (%) on
five generalization benchmarks. The column ‘∆Avg.’ high-
lights the performance gain attributable to the RAG module
within each stage. Crucially, we observe that RAG provides
substantial improvements consistently across all stages,
demonstrating that dynamic retrieval of expert knowledge is
fundamental to bridging the generalization gap, even in the
early visual alignment phase.
Stage ConfigTest Set AUC∆CDF1 CDF2 DFDC FFIW WDF
Stage-1w/oRAG 89.72 86.30 73.31 78.59 78.73+8.16w/ RAG 96.77 94.03 78.54 90.89 87.63
Stage-2w/oRAG 91.33 86.96 71.57 76.43 70.05+10.66w/RAG 91.53 95.80 81.58 92.37 89.13
Stage-3w/oRAG 91.53 86.98 72.05 76.51 70.04+12.2w/RAG 99.6 95.97 81.82 93.49 88.96
Table 9. Effectiveness across different base MLLMs (AUC
%). We show the performance of each model before (zero-
shot) and after (’+ VRAG-DFD (Ours)’) applying our three-
stage VRAG-DFD pipeline.
MetricsTest Set AUC
CDF1 CDF2 DFDC FFIW WDF
Qwen2.5VL-7B 66.00 73.35 56.82 71.91 66.82
Qwen2.5VL-7B + ours 99.60 95.97 81.82 93.49 88.96
Intern3VL-8B 66.33 72.06 57.19 70.86 65.75
Intern3VL-8B + ours 99.19 95.09 80.16 92.45 90.00
framework, consistently bridging the knowledge gap
from initial visual alignment to final critical reason-
ing.
C.3. Effectiveness on Different Base MLLMs.
To verify the model-agnostic nature of the VRAG-
DFD framework, we replace the defaultQwen2.5VL
withInternVL3-8B, applying our full three-stage
training pipeline to each. As summarized in Table 9,
all base models exhibit substantial improvements in
cross-dataset detection performance after training un-
der our framework. This demonstrates that our ap-
proach serves as a general and effective paradigm for
teaching MLLMs advanced forensic reasoning skills.
C.4. Latency Analysis of the RAG Module
The construction and indexing of the FKD database
are performed entirely offline, ensuring that no addi-
tional inference overhead is introduced during deploy-
ment. During inference, the RAG module only per-
forms FAISS-based retrieval over precomputed fea-
ture embeddings. As shown in Tab. 10, the retrieval
operation accounts for only 0.35% of the total compu-
tational cost. This negligible overhead indicates that
the integration of the RAG module has a minimal im-
pact on overall inference latency.

Table 10. Computational complexity analysis of VRAG-
DFD.
Retrieval Inference Total
Metric GFLOPs Ratio GFLOPs Ratio GFLOPs Ratio
Value∼810.35%∼22,760 99.65%∼22,841 100.00%
Table 11.Ablation study on frame sampling density in
Stage-1.We compare sparse sampling (8 frames/video)
against dense sampling (all frames). The results demon-
strate asubstantial performance leapin the average AUC
(+3.80%), confirming that dense visual supervision is indis-
pensable for learning robust forgery features across diverse
datasets.
Samp. Strategy TotalTest Set AUCAvg.CDF1 CDF2 DFDC FFIW WDF
Sparse (8 fms) ∼28k 92.1476.85 70.69 70.40 77.95 77.61
Dense (All fms) ∼80k 89.7286.30 73.71 78.59 78.73 81.41
Gain (∆) - -2.42+9.45 +3.02 +8.19 +0.78 +3.80
C.5. Impact of Frame Sampling Density in
Stage-1
In Stage-1, we adopt dense frame sampling over all
2,500 training videos (approximately 80k images) to
maximize the model’s visual alignment capability. To
validate the necessity of this computationally intensive
strategy, we compare it with sparse sampling, where
only 8 frames per video are selected.
Results.As reported in Table 11, dense sam-
pling significantly outperforms sparse sampling, in-
creasing the average AUC from 77.61% to 81.41%,
corresponding to a performance gain of nearly +3.8%.
Analysis.This substantial improvement reveals
two key insights:
1.Necessity of Dense Supervision.Forgery arti-
facts in deepfakes, such as frame-to-frame jitter
and transient texture anomalies, are often sparse
and non-continuous. Sparse sampling may miss
these critical discriminative frames, leading to an
incomplete feature space.
2.Solid Foundation for Subsequent Stages.Stage-
1 visual alignment serves as the cornerstone of
the entire VRAG-DFD framework. Our experi-
ments indicate that dense training with all available
frames is essential to obtain a sufficiently robust
visual encoder, which underpins the reasoning-
centric Stage-2 and Stage-3.
C.6. Robustness against Adversarial Retrieval
Noise
To evaluate whether VRAG-DFD can resist mislead-
ing external evidence, we specifically analyze adiffi-
cult subsetof the test set. This subset is defined as
samples where the retriever fails, i.e., the majority of
the top-kretrieved evidences belong to incorrect cate-
gories.
Results.As shown in Table 12, VRAG-
DFD demonstrates remarkable resilience even underTable 12.Robustness against misleading retrieval evi-
dence.We evaluate a subset of adversarial samples where
the initial visual perception is correct (s 1=Correct), but the
RAG evidence is misleading (RAG=False). The results
demonstrate that VRAG-DFD maintains an exceptionally
highRobustness Rate(>96%on avg.), effectively reject-
ing misleading external noise.
MetricTest Set AUCWeigh. Avg.CDF-v1 CDF-v2 DFDC FFIW WDF
# Adv. Samp. 34 429 534 1,374 5,226 7,597
# VRAG DFD Cor. 32 417 520 1,345 5,048 7,362
Robust. Rate (%) 94.12 97.20 97.38 97.89 96.59 96.91
strongly misleading retrieval conditions. Across all
five evaluation datasets, the proportion of correctly
classified samples consistently exceeds 94%.
Analysis.These results indicate that VRAG-DFD
does not simply “copy” the retrieved evidence. In-
stead, it successfully learns to critically weigh the reli-
ability of external evidence and perform reasoning that
accounts for potential inconsistencies, highlighting its
robust and principled inference capabilities.
C.7. Qualitative Analysis
To visually demonstrate the interpretability and rea-
soning capabilities of VRAG-DFD, we present qual-
itative examples in Figures 5, 6, 7 and 8. These fig-
ures display the query images alongside the complete
Forensic Chain-of-Thought (F-CoT) responses gener-
ated by our model, illustrating how it integrates visual
analysis and retrieval evidence to reach a final verdict.

Real<Preliminary Visual Analysis>
Upon examining the ` Query_Image `, I observe several key visual characteristics. The overall image quality is low, but the facial features appear 
consistent with the image's resolution. The lighting on the face is coherent, with plausible shadows under the chin and on th e right side of the face, 
suggesting a single light source. The skin texture, while not high -resolution, is uniform across the entire face, including the forehead, cheeks, and chin. 
There are no isolated patches of unnatural smoothness or blurring . The transition between the face and the hairline, as well as the jawline, appears 
natural without any sharp, artificial edges or smudging artifacts. The facial features, such as the eyes and mouth, are proportionally aligned and do not 
show obvious signs of anatomical distortion.
</Preliminary Visual Analysis>
<RAG Reference Information Analysis >
The retrieval report provides the following information:
Supporting Evidence (supports my judgment) :`(\"Real: [Consistent Skin Texture]: The overall level of blurriness is consistent across the entire face, 
hair, and background, which is a characteristic of an authentic, unedited low -resolution image. \", 0.692)`
Opposing Evidence (contradicts my judgment) : `(\"Fake: [Mouth]: The teeth are unnaturally shaped and misaligned within the mouth, appearing 
distorted and blocky. \", 0.717)`, `( \"Fake: [Eyes]: The eyelids and periocular areas exhibit significant blurriness, lacking the pores and fine 
textures... \", 0.704)`
Analysis :The RAG report is noisy and presents a significant conflict. It contains a very high -scoring `Fake` reference (0.717) that points t o specific 
artifacts, but also a high -scoring `Real` reference (0.692) that suggests the image is authentic due to consistent low resolutio n. A careful cross -
verification is required.
</RAG Reference Information Analysis>
<Fusion, Reasoning, and Decision>
My preliminary visual analysis indicates that the image displays characteristics of an authentic , low -resolution photograph. I must now cross -verify 
this against the conflicting RAG report:
1. Verify Opposing Evidence : The RAG report's highest -scoring `Fake` evidence (similarity 0.717) claims \"unnaturally shaped and misaligned teeth \" 
and \"significant blurriness \" in the eyes. Upon re -examining the ` Query_Image `, I cannot confirm these specific artifacts . The teeth are not clearly 
defined , but their shape is consistent with the overall low resolution of the image, not a specific manipulation . The blurriness in the eyes is consistent 
with the blurriness of the rest of the face, which is a sign of low image quality, not a localized forgery artifact. Therefor e, this evidence is likely 
retrieval noise from a semantically similar but genuinely fake image and must be discarded .
2. Verify Supporting Evidence :The RAG report's `Real` evidence (similarity 0.692) states that \"the overall level of blurriness is consistent across the 
entire face, hair, and background. \" This observation can be confirmed and perfectly aligns with my visual analysis. The consist ency of the low 
resolution across all facial features is a strong indicator of authenticity .Conclusion : My preliminary visual analysis, which identified consistent low -
resolution features, has been strongly cross -verified by the \"useful evidence \" from the RAG report. The high -scoring \"noise e vidence \" from RAG, 
which suggested forgery, has been effectively refuted by the visual facts of the ` Query_Image `. Therefore, my final judgment is that the image is Real.
</Fusion, Reasoning, and Decision>
<answer> Real </answer>Figure 5. Qualitative analysis of VRAG-DFD (real case 1).
<Preliminary Visual Analysis>
Upon examining the ` Query_Image `, I observe several key visual characteristics. The skin texture on the man's face, particularly on the cheeks and 
forehead, appears consistent with the overall low resolution of the image. There are no isolated patches of unnatural smoothness or blurring. The 
lighting across the face seems coherent, with shadows and highlights that are consistent with a single light source. The faci al features, such as the 
furrowed eyebrows and the expression of the mouth, appear anatomically plausible and coordinated. The transition from the jaw line to the neck and 
the hairline to the forehead is smooth, without any sharp, artificial edges or color mismatches. The stubble on the chin and around the mouth retains 
some fine texture, which is consistent with the rest of the facial hair. </Preliminary Visual Analysis>
<RAG Reference Information Analysis >
The retrieval report provides the following information:
Supporting Evidence (supports my judgment):`( \"Real: [Consistent Skin Texture]: Although the image has low resolution, the degree of softness and 
lack of fine detail on the facial skin is consistent with the blurriness of the hair and the background, indicating the entir e image was captured with the 
same focus and quality. \", 0.6406)`
Opposing Evidence (contradicts my judgment):`( \"Fake: [Mouth]: The lips appear blurry and lack the fine lines and texture... [Skin]: The skin on the 
chin and around the mouth is unnaturally smooth \", 0.6383)`, `( \"Fake: [Mouth]: The lips in the manipulated image lack sharp def inition and fine 
wrinkles, appearing blurry and overly smooth. \", 0.6308)`
Analysis: TheRAG report is noisy and presents a significant conflict. It provides high -scoring evidence for both `Real` (0.64) and `Fake` (0 .64) labels, 
with the scores being very close. This requires careful cross -verification with the primary visual evidence.
</RAG Reference Information Analysis>
<Fusion, Reasoning, and Decision>
My preliminary visual analysis indicates that the image displays characteristics of authenticity, such as consistent texture and plausible facial features. 
I must now cross -verify this against the conflicting RAG report :
1. Verify Opposing Evidence :The RAG report's highest -scoring `Fake` evidence (similarity 0.6383) claims the \"lips appear blurry \" and the \"skin on 
the chin and around the mouth is unnaturally smooth. \" This assertion cannot be confirmed by my visual inspection. The blurriness of the lips is 
consistent with the overall low resolution of the entire image, not an isolated artifact. The skin texture on the chin and around the mouth is not 
unnaturally smooth; it is consistent with the texture on the forehead and cheeks. Therefore, this evidence is likely retrieval noise, mistaking low image 
quality for a forgery artifact, and must be discarded.
2. Verify Supporting Evidence : The RAG report's `Real` evidence (similarity 0.6406) states that \"the degree of softness and lack of fine detail on the 
facial skin is consistent with the blurriness of the hair and the background. \" This statement can be confirmed and perfectly aligns with my visual 
analysis. The consistency of the low -resolution effect across the entire image is a strong indicator of authenticity . Conclusion: My preliminary visual 
analysis suggesting the image is `Real` has been strongly cross -verified by the credible \"Supporting Evidence \" from the RAG re port. The high -
scoring \"Opposing Evidence \" has been effectively refuted and rejected as it contradicts the visual facts of the `Query_ Image `. Therefore, based on 
the primary visual evidence, reinforced by the verified RAG context, my final judgment is that the image is authentic.
</Fusion, Reasoning, and Decision>
<answer> Real </answer>Real
Figure 6. Qualitative analysis of VRAG-DFD (real case 2).

<Preliminary Visual Analysis>
Upon examining the ` Query_Image `, I observe several key visual characteristics . The skin texture on the man's face, particularly across the forehead 
and cheeks, appears unusually smooth and lacks the fine details typical of natural skin , such as pores or fine lines. This smoothness gives the face a 
slightly \"waxy \" or \"mask -like\" quality. Additionally, the area around the mouth, specifically the lips, seems to have a subtle blurriness and lacks 
sharp definition . The transition between the lips and the surrounding skin is not as crisp as one might expect.
</Preliminary Visual Analysis>
<RAG Reference Information Analysis >
The retrieval report provides the following information:
Supporting Evidence (supports my judgment): `(\"Fake: [Skin]: The skin on the manipulated face , especially in the forehead and cheek areas, appears 
unusually smooth, lacking high -frequency details... This detail loss phenomenon gives the face a 'mask -like' appearance. \", simi larity: 0.5938)`  
Opposing Evidence (contradicts my judgment): `(\"Real: [Consistent Skin Texture]: ...the level of blurriness and digital noise on the facial skin is 
consistent with that of the neck, hair, and background... \", similarity: 0.5938)` 
Analysis: TheRAG report is highly conflicting. The top -scoring reference (0.5938) suggests the image is `Fake` due to unnatural skin smoothn ess, 
while another reference with the same score suggests it is `Real` due to consistent blurriness. This indicates significant no isein the retrieval, requiring 
careful cross -verification with the primary visual evidence.
</RAG Reference Information Analysis>
<Fusion, Reasoning, and Decision>
My preliminary visual analysis identified unnaturally smooth skin and a lack of definition around the mouth . I must now cross -verify this against the 
conflicting RAG report:
1. Verify Opposing Evidence: The RAG report's `Real` evidence (similarity 0.5938) claims that the blurriness is consistent across the entire image. 
However, this claim cannot be fully confirmed by my visual inspection. While the entire image is low -resolution, the smoothness on the face appears 
more pronounced and lacks the fine texture that would be expected, even in a blurry image. The skin on the forehead and cheek s has a different 
quality compared to the neck and clothing. Therefore, this evidence is likely retrieval noise, mistaking general low quality for consistent authenticity , 
and must be discarded.
2.Verify Supporting Evidence : The RAG report's `Fake` evidence (similarity 0.5938) states that the skin \"appears unusually smooth, lacking high -
frequency details \" and has a \"'mask -like' appearance .\" This description can be directly confirmed on the ` Query_Image `. The waxy, textureless
quality I observed on the man's forehead and cheeks perfectly matches this description. The two pieces of evidence are mutually reinforcing.
Conclusion:  My preliminary visual analysis, which pointed towards forgery due to unnatural skin texture, has been strongly c ross-verified by the 
useful evidence from the RAG report. The conflicting \"noise \" evidence from RAG has been effectively rejected by the visual facts. Therefore, based 
on the primary visual evidence, reinforced by the credible RAG supporting evidence, my final judgment is that the image is Fa ke.
</Fusion, Reasoning, and Decision>
<answer> Fake </answer>FakeFigure 7. Qualitative analysis of VRAG-DFD (fake case 1).
<Preliminary Visual Analysis>
Upon examining the ` Query_Image `, I observe several key visual characteristics. The skin texture on the woman's face , particularly on the cheeks and 
forehead, appears unusually smooth and lacks the fine , high -frequency details one would expect in a natural photograph, even a low -resolution one. 
There is a noticeable absence of natural wrinkles or pores. Additionally, the area around the mouth and nose seems slightly blurry and lacks sharp 
definition, which contrasts with the relative clarity of the hair strands. The lighting on the face seems consistent, but the texture itself feels artificially 
uniform and flat. \n
</Preliminary Visual Analysis>
<RAG Reference Information Analysis >
The retrieval report provides the following information:
Supporting Evidence (supports my judgment):`( \"Fake: [Skin]: The skin on the cheeks and around the mouth appears abnormally smooth and blurry, 
with a noticeable absence of fine wrinkles and natural skin texture. \", 0.650)`, `(\"Fake: [Mouth]: The lip region lacks sharpness and fine details, such 
as natural lip creases, making it appear slightly blurry. \", 0.650)`
Opposing Evidence (contradicts my judgment): `(\"Real: [Consistent Skin Tone/Lighting]: The lighting on the face is consistent, originating from the 
front -left. This creates plausible shadows on the right side of her face and under her nose. \", 0.670)`
Analysis: The RAG report is noisy and presents a significant conflict. It contains a very high -scoring piece of opposing evidenc e (0.670) that suggests 
the image is `Real`, but also provides high -scoring supporting evidence (0.650) that points to forgery. A careful cross -validati on is required. \n
</RAG Reference Information Analysis>
<Fusion, Reasoning, and Decision>
My preliminary visual analysis identified unnaturally smooth skin and a lack of fine details around the mouth. I must now cro ss-verify this against the 
conflicting RAG report:
1. Verify Opposing Evidence : The RAG report's highest -scoring `Real` evidence (similarity 0.670) claims \"Consistent Skin Tone/Lighting \". While 
the lighting is indeed consistent, this claim cannot be fully confirmed because it overlooks the key visual artifact I identified: the complete absence of 
natural skin texture. The claim of \"consistent \" skin tone is misleading because the texture is not natural. Therefore, this ev idence is likely retrieval 
noise based on semantic similarity (a woman with glasses) rather than a correct analysis of forgery artifacts , and it must be discarded.
2. Verify Supporting Evidence : The RAG report's `Fake` evidence (similarity 0.650) states that the skin is \"abnormally smooth and blurry \" and lacks 
\"fine wrinkles and natural skin texture. \" This description can be directly confirmed and perfectly matches my visual observati on of the 
`Query_Image `. The lack of texture is a significant indicator of manipulation.
Conclusion: My preliminary visual analysis, which pointed towards a `Fake` image due to unnaturally smooth skin, has been str ongly cross -verified 
by the useful evidence from the RAG report. The high -scoring opposing evidence has been rejected as it contradicts the clear vis ual facts present in 
the image. Therefore, based on the primary visual evidence, reinforced by the credible RAG supporting evidence, my final judg ment is that the image 
is Fake.
</Fusion, Reasoning, and Decision>
<answer> Fake </answer>Fake
Figure 8. Qualitative analysis of VRAG-DFD (fake case 2).

Prompt for DeepFakes Annotation
System Prompt:
You are an expert in face tampering detection. Your task is to strictly compare the[Manipulated
Image]and the[Original Image], first precisely locate the manipulated regions, and thenexplain
how the forgery artifacts within those regions were caused by the improper processing of DeepFakes tech-
nology. You must strictly adhere to the “Based on Comparison, Loyal to Evidence” principle.Hallucina-
tions are strictly forbidden. Focus on artifacts left by theforgery process(e.g., blending, alignment), not
natural facial features (e.g., makeup, appearance). Your output must strictly follow the “Output Format”,
listing manipulated regions first, then analyzing the artifacts one by one.
User Prompt:
Task Definition:You will receive two images: the first is a[Manipulated Image]generated using
DeepFakes technology, and the second is the corresponding[Original Image]. Carefully compare
the two images to locate manipulated regions, then explain the forgery artifacts using the “DeepFakes
Potential Forgery Artifacts Reference Guide”. Focus solely on artifacts caused by DeepFakes, not natural
differences.
Core Analysis Principles:
1. Carefully compare all facial features, hair, skin color, etc., to locate manipulated regions without omis-
sions or hallucinations.
2. Distinguish between:
•Analyze This - Forgery Artifacts: Anomalies caused by technical flaws (e.g., low-resolution bottle-
necks, blending, alignment failures).
Correct Example: Inconsistent skin tone, feature overlap, blurry edges, structural distortion.
•Ignore This - Natural Features: Inherent facial features unrelated to forgery.
Incorrect Example: Makeup, moles, dimples, natural face shape or expression.
3. Attribute observed phenomena (e.g., “skin looks airbrushed”) to the technical flaw (e.g., “detail loss
due to low resolution”).
4. Report all types of artifacts present in a region, not only the most obvious.
DeepFakes Potential Forgery Artifacts Reference Guide:
1.Blending Border Artifacts (High-Incident Area)
•Cause: Poisson blending at facial edges (face periphery, hair, chin, neck).
•Manifestation: Edge color differences, unnatural blurring/halo effects along contours.
2.Structural Abnormality (Feature Misalignment/Distortion)
•Cause: Keypoint alignment failure.
•Manifestation: Distorted or misaligned facial features; feature overlap (e.g., eyebrows).
3.Detail Loss
•Cause: Low-resolution face generation.
•Manifestation: Blurry eyes, teeth, irises; abnormally smooth skin lacking texture or pores.
Output Format:
First list manipulated regions, then describe the corresponding forgery artifacts.
Reference Example:
Manipulated Regions: Skin, Eyebrows
Forgery Artifacts: [Skin]: Central area cool white, periphery yellowish-black, clear blending boundary.
[Eyebrows]: Ghosting caused by facial alignment failure.
Here are[Manipulated Image]and the[Original Image].
[Manipulated Image]:{{manipulated image path}}
[Original Image]:{{original image path}}

Prompt for Face2Face Annotation
System Prompt:
You are an expert in face tampering detection. Your task is to strictly compare the[Manipulated
Image]and the[Original Image], first precisely locate the manipulated regions, and thenexplain
how the forgery artifacts within those regions werecaused by the Face2Face 3D model re-rendering
process. You must strictly adhere to the “Based on Comparison, Loyal to Evidence” principle.Halluci-
nations are strictly forbidden. Focus on artifacts left by theforgery process(e.g., 3D model tracking,
re-rendering, expression transfer), not natural facial features (e.g., makeup, appearance). Your output must
strictly follow the “Output Format”, listing manipulated regions first, then analyzing the artifacts one by
one.
User Prompt:
Task Definition:You will receive two images: the first is a[Manipulated Image]generated using
Face2Face technology, and the second is the corresponding[Original Image]. Carefully compare
the two images to locate manipulated regions, then explain the forgery artifacts using the “Face2Face
Potential Forgery Artifacts Reference Guide”. Focus solely on artifacts caused by Face2Face technology,
not natural expression differences.
Core Analysis Principles:
1. Identify all differences to locate manipulated areas, including facial features, hair, and skin tone. Pay
special attention to the lips and surrounding areas.
2. Distinguish between:
•Analyze This - Forgery Artifacts: Technical anomalies (3D re-rendering, Blendshape parameterized
driving, 3D tracking errors).
Correct Example: ’Plastic-like’ skin, expression structural distortion (’puppet-like’), misalignment at
facial edges, unrealistic lighting/highlights.
•Ignore This - Natural Features: Expression changes themselves are not artifacts.
Incorrect Example: Different expressions (smiling vs laughing), mouth open, [Manipulated Image]
laughing while [Original Image] is smiling.
3. Base your analysis on the technical flaws outlined in the Reference Guide; attribute observed phenom-
ena to the correct cause.
4. Note: Face2Face preserves identity; focus on expression manipulation artifacts, not identity differences.
Face2Face Potential Forgery Artifacts Reference Guide:
1.3D Model Render/Blend Borders
•Cause: Re-rendered facial regions pasted onto the original frame.
•Manifestation: Facial edge mismatch, background/hair distortion adjacent to the face.
2.Texture Mismatch Due to Re-rendering
•Cause: 3D model cannot perfectly replicate original camera imaging details or skin textures.
•Manifestation: ’Plastic-like’ skin, unrealistic lighting/highlights.
3.Local Artifact: Mouth Detail Blurring
•Cause: Difficulty preserving fine lip textures during expression manipulation.
•Manifestation: Loss of lip texture, blurry lip borders, stiff lip shape.
4.Expression Structural Distortion (”Puppet-like” Artifacts)
•Cause: Parameterized Blendshape coefficients limit realistic muscle coordination.
•Manifestation: Non-ergonomic stretching, stiff/mechanical expressions, lack of natural coordination
between mouth and eyes.
Output Format:
First list manipulated regions, then describe corresponding forgery artifacts.
Reference Example:
Manipulated Regions: Facial Contour, Mouth
Forgery Artifacts: [Facial Contour]: Blending artifacts at the contour. [Mouth]: Smile shape stiff and
unnatural; corners stretch mechanically.
Here are[Manipulated Image]and the[Original Image].
[Manipulated Image]:{{manipulated image path}}
[Original Image]:{{original image path}}

Prompt for FaceSwap Annotation
System Prompt:
You are an expert in face tampering detection. Your task is to strictly compare the[Manipulated Image]and the
[Original Image], first precisely locate the manipulated regions, and thenexplainhow the forgery artifacts within
those regions werecaused by the improper processing of FaceSwap technology. You must strictly adhere to the “Based
on Comparison, Loyal to Evidence” principle.Hallucinations are strictly forbidden. Your analysis of the manipulated
regions must focus on the artifacts left by theforgery process(e.g., blending, alignment), not the natural features of the
face (e.g., makeup, appearance). Your output muststrictly followthe “Output Format” specified by the user, listing the
manipulated regions first, then analyzing the artifacts one by one.
User Prompt:
Task Definition:You will receive two images, the first is a[Manipulated Image]generated using FaceSwap tech-
nology, and the second is the corresponding[Original Image]. Please act as an expert in face tampering detection.
By carefully comparing the two images, first find the manipulated regions in the[Manipulated Image], and then
explainthe forgery artifacts in those regions. You must first locate the manipulated regions by careful comparison, and
then use the “FaceSwap Potential Forgery Artifacts Reference Guide” toexplainhow the artifacts you found were caused
by FaceSwap processing. Focus on the artifacts caused by FaceSwap, not natural facial differences.
Core Analysis Principles:
1. Carefully compare each facial feature, hair, skin, etc., to locate manipulated regions without omissions or hallucina-
tions.
2. Distinguish clearly between:
•Analyze This - Forgery Artifacts: Visual anomalies directly caused by technical flaws (e.g., blending, alignment,
3D fitting).
Correct Example: Inconsistent skin tone, feature overlap, blurry edges, structural distortion.
•Ignore This - Natural Features: Inherent facial features unrelated to forgery.
Incorrect Example: Makeup, moles, dimples, natural face shape or expression.
3. Attribute observed phenomena (e.g., “two eyebrows”) to technical flaws (e.g., “feature overlap caused by keypoint
mismatch”).
FaceSwap Potential Forgery Artifacts Reference Guide:
1. Color & Lighting Inconsistency
•Inconsistent Skin Tone: Skin hue, saturation, or brightness differs from surrounding regions.
•Incorrect Lighting/Shadows: Face highlights/shadows inconsistent with environment.
2. Feature Misalignment
•Facial Structure Abnormality: Misaligned facial features due to sparse keypoint fitting.
•Feature Overlap: Overlapping features from both faces (e.g., eyebrows).
3. Blending Artifacts
•Unnatural Blurring: Intentional blur to hide splicing lines.
•Edge Overlap/Gaps: Misalignment with head contour causing gaps or overlaps.
•Hard Edges: Clear cutting sensation from poor blending.
4. Mask-like Feel
•Detail Loss: Overly smooth/blurry skin creating a mask-like appearance.
Output Format:
First list the manipulated regions, then describe the corresponding forgery artifacts.
Reference Example:
Manipulated Regions: Eyebrows, Skin
Forgery Artifacts: [Eyebrows]: Ghosting due tofeature overlap. [Skin]: Cheek tone mismatch exposingblending defect.
Here are[Manipulated Image]and the[Original Image]:
[Manipulated Image]:{{manipulated image path}}
[Original Image]:{{original image path}}

Prompt for NeuralTextures Annotation
System Prompt:
You are an expert in face tampering detection. Your task is to strictly compare the[Manipulated
Image]and the[Original Image], first precisely locate the manipulated region (entire face), and
thenexplainhow the forgery artifacts werecaused by the NeuralTextures GAN-based full-face re-
rendering process. You must strictly adhere to the “Based on Comparison, Loyal to Evidence” principle.
Hallucinations are strictly forbidden. Focus on artifacts left by theforgery process(e.g., loss of high-
frequency details), not natural facial features. Your output must strictly follow the “Output Format”, listing
manipulated regions first, then analyzing the artifacts one by one.
User Prompt:
Task Definition:You will receive two images: the first is a[Manipulated Image]generated us-
ing NeuralTextures technology, and the second is the corresponding[Original Image]. Carefully
compare the two images to locate manipulated regions (full face) and explain the forgery artifacts using
the “NeuralTextures Potential Forgery Artifacts Reference Guide”. Focus solely on artifacts caused by
NeuralTextures technology.
Core Analysis Principles:
1. Identify all differences across the full face, including skin texture, wrinkles, smile lines, and lips. Pay
special attention to the lips and surrounding areas.
2. Distinguish between:
•Analyze This - Forgery Artifacts: Pixel-level anomalies caused by GAN re-rendering.
Correct Example: “Airbrushed” feel, loss of high-frequency details (pores, wrinkles, smile lines),
loss of lip texture, blurry lip borders.
•Ignore This - Semantic Changes/Natural Features: Expression changes are not artifacts.
Incorrect Example: Open mouth, smiling vs laughing.
3. Attribute observed phenomena to technical flaws according to the Reference Guide.
4. Note: NeuralTextures preserves identity; focus on expression and full-face re-rendering artifacts.
NeuralTextures Potential Forgery Artifacts Reference Guide:
1.Core Artifact: Global High-Frequency Detail Loss
•Cause: GAN/U-Net reconstruction of the entire face smooths out high-frequency details.
•Manifestation:
–“Airbrushed” feel across the entire face (forehead, cheeks, nose).
–Loss of skin texture, pores, smile lines, crow’s feet; details are faded or missing.
2.Local Artifact: Mouth Detail Blurring
•Cause: GAN may struggle to preserve fine lip textures.
•Manifestation: Loss of lip texture, blurry lip borders, stiff or flat lip shape.
Output Format:
First list manipulated regions, then describe corresponding forgery artifacts.
Reference Example:
Manipulated Regions: Facial Skin, Lips
Forgery Artifacts: [Facial Skin]: Entire face looks abnormally smooth, “airbrushed”, lacks pores and
texture. [Lips]: Lip surface abnormally smooth, missing lip texture.
Here are[Manipulated Image]and the[Original Image].
[Manipulated Image]:{{manipulated image path}}
[Original Image]:{{original image path}}

Prompt for Real Image Annotation
System Prompt:
You are an expert in face tampering detection. Your task is to strictly analyze the[Real Image]and“provide evi-
dence”of its authenticity. You must accomplish this task byselectivelyconfirming that the imagepossesses “Indicators
of Authenticity”(e.g., normal facial structure) andlacks “Forgery Artifacts”(e.g., no splicing lines on the face). Your
analysis must bebased on facts: if a real, low-resolution image is blurry, youmust notlabel “clear pores,” but should
instead focus on analyzing other indicators like “lighting consistency” or “facial structure”. Your output muststrictly
followthe “Output Format” specified by the user.
User Prompt:
Task Definition:
You will receive a[Real Image]. Please act as an expert in face tampering detection toanalyzeandexplainwhy
this image is real. Your goal is to teach a downstream model to recognizeIndicators of Authenticity. Hallucinations are
strictly forbidden during this process. If you do not see obvious real features, you should check if other real features exist
and must not fabricate them.
Core Analysis Principles (Important):
1.Selective Annotation (Key Principle):The “Indicators of Authenticity Reference Guide” is a“checklist,”not a
“requirement”.
• Youmustselectonlythose features from the guide that areclearly and distinctly visiblein this[Real Image].
•No Hallucinations: If a real, low-resolution photo causes “skin texture” to be blurry, youmust notselect
[Skin Texture]. Skip it and analyze reliable indicators like[Skin Tone Consistency]or[Facial
Structure].
2.Analysis Order (From Easy to Hard):
•First (Check Face Swap):Ensure obvious artifacts (feature structure, contour edges, skin tone) areabsent.
•Second (Check Reenactment):Ensure subtle artifacts (skin texture, lip texture/shape) areabsent.
3.Core Difference (Consistency):Forgery =Inconsistent; Real =Consistent. Your analysis should reflect this consis-
tency.
Indicators of Authenticity Reference Guide:
(Select and analyze only clearly observable features)
1.Structure & Contour (vs. FaceSwap/DeepFakes flaws)
•Normal Facial Structure: Features do not show misalignment, distortion, or overlap.
•Natural Contour Transitions: Smooth transitions to neck/hair;nosplicing artifacts, hard edges, or un-
natural blurring.
2.Lighting & Color (vs. FaceSwap/DeepFakes flaws)
•Consistent Skin Tone/Lighting: Base skin tone and lighting direction are consistent across face/neck.
No color patches.
•Natural Shadows/Highlights: Physically plausible shape and softness of shadows/highlights.
3.Geometric Shape (vs. Face2Face/NT flaws)
•Natural Lip Shape: Lips have curvature;lacksstiff, over-smoothed, or puppet-like feel.
•Coordinated Facial Muscles: Muscle movements (e.g., smile lines) follow anatomical logic.
4.High-Frequency Details (Depends on image clarity)
•Consistent Skin Texture:(Clear Image)Pores/wrinkles visible and uniform.(Blurry Image)Blur level
is consistent with background.
•Clear Lip Texture:(Clear Image)Visible texture and natural sheen; clear borders.
•Hair/Teeth Details:(Clear Image)Sharp strands of hair and teeth texture.
Output Format:
Reference Example 1: High-Quality Image
Indicators of Authenticity:
[Skin Texture Details]:The skin texture on the cheeks and forehead is clear and consistent with the neck texture; under-
eye bags are visible.
[Lip Texture]:The lip texture is clear and natural, with a natural sheen.
[Lighting Consistency]:The lighting direction on the face and neck is consistent, with no local anomalies.
Reference Example 2: Low-Quality/Blurry Image
Indicators of Authenticity:
[Facial Structure]:The facial features conform to anatomical structure, with no misalignment or overlap.
[Contour Transitions]:Although the image is blurry, the transition from the facial contour to the background is still
natural, with no splicing lines.
[Lip Shape]:The lip shape is natural and not stiff; the shadow under the lower lip is natural.
Here are the[Real Image].
[Real Image]:{{real image path}}

Prompt for Cross-Validation Chain-of-Thought
System Prompt:
Task Definition:
You are a world-class face tampering detection analyst and an expert in logical reasoning. Your task is to “role-play” and
generate aCross-Validation Chain-of-Thoughtto create fine-tuning data. You will receive:
1.Query Image(Primary Evidence)
2.RAG Context(Secondary Evidence: 5 retrieved reference annotations + similarity scores, possibly noisy)
3.Ground Truth Label
Your reasoning must follow three steps:
1. Preliminary Visual Analysis:• Analyze theQuery Imagein detail.
• Describe only the visual facts that indicate support for theGround Truth Label(e.g., “skin color on the face
is uniform”, “an unnatural blur at the mouth’s edge”).
• Do not mention or assume the Ground Truth. Only report visual evidence.
2. RAG Reference Information Analysis:• Objectively report the 5 pieces of retrieved information.
• Identify:
– Supporting Evidence (consistent with Step 1): e.g., (”Real: clear skin texture”, 0.70)
– Opposing Evidence (contradicts Step 1): e.g., (”Fake: mouth artifact”, 0.85)
• Provide an objective analysis of the RAG context, noting both support and noise.
3. Fusion, Reasoning, and Decision:• Cross-verify visual facts from Step 1 against RAG evidence:
1.Verify Opposing Evidence:If RAG’s Fake evidence cannot be confirmed visually, treat it as retrieval noise
and discard it.
2.Verify Supporting Evidence:If RAG’s Real evidence can be confirmed visually, accept it as valid.
• Conclude by integrating Step 1 and Step 2:
–Visual analysis supports the Ground Truth.
–Supporting RAG evidence reinforces the conclusion.
–Opposing RAG evidence is discarded as noise.
Output Format:
<Preliminary Visual Analysis>:
[Analyze the Query Image in detail. Only describe visual facts that lead to theGround Truth Label.]
</Preliminary Visual Analysis>
<RAG Reference Information Analysis>:
[”The retrieval report provides the following information:
Supporting Evidence:[Cite 1-2 highest-scoring pieces consistent with Step 1]
Opposing Evidence:[Cite 1-2 highest-scoring pieces contradicting Step 1]
textbfAnalysis: Objectively evaluate the RAG context. ”]
</RAG Reference Information Analysis>
<Fusion, Reasoning, and Decision>:
[Step 1 visual facts + cross-verification with Step 2:
1. Verify Opposing Evidence→discard if inconsistent
2. Verify Supporting Evidence→confirm if consistent
3. Conclusion: combine visual facts and valid RAG evidence to form final judgment.]
</Fusion, Reasoning, and Decision>
<Answer> Ground Truth Label </Answer>

Prompt for Evidence-Guided Correction Chain-of-Thought
System Prompt:
You are a world-class face tampering detection analyst and an expert in logical reasoning. Your task is to “role-play” and
generate aCross-Validation Chain-of-Thoughtto create fine-tuning data. You will receive:
1.Query Image(Primary Evidence)
2.RAG Context(Secondary Evidence: 5 retrieved reference annotations + similarity scores)
3.Ground Truth Label
You must act as a perfect student model that initially misjudges the image but allows the RAG report to correct its
intuition. Your reasoning must follow three steps:
1. Visual Analysis (Admit Confusion):Analyze theQuery Imagein detail. Pretend to be misled by confusing
features and form a wrong initial impression. Do not mention theGround Truth Label.
2. RAG Analysis (Highlight Golden Evidence):Inspect theRAG Context. Identify the “Golden Evidence” con-
sistent with theGround Truth Label(e.g., high-score>0.8, dominant quantity). Only report RAG contents
without making a conclusion.
3. Fusion and Decision (The “Aha!” Moment):Cross-verify RAG evidence and correct the initial visual intuition.
Steps:
1.Re-examination:Re-inspect theQuery Imageprompted by Golden Evidence.
2.Verification (Aha! Moment):Describe subtle artifacts now visible due to RAG guidance.
3.Conclusion:Declare the final judgment after RAG-guided re-evaluation.
Output Format:
<Preliminary Visual Analysis>:
Describe the confusing visual features that may lead to a wrong initial judgment. E.g., “I (the student) observe that the
overall lighting of theQuery Imageseems consistent, and no obvious artifacts are immediately visible to the naked
eye. At first glance, it seems to be a [Wrong Label]. ”
</Preliminary Visual Analysis>
<RAG Reference Information Analysis>
“However, the RAG report provides high-confidence ‘Golden Evidence’ that contradicts my initial impression:
1.Golden Evidence (Corrective):[Cite the 1-2 highest-scoring pieces of evidence consistent with the GT, e.g., (“Fake:
subtle mouth artifact”, 0.91)]
2.Majority Consensus:[Point out quantity distribution, e.g., “Among the 5 retrieved pieces, 4 consistently point to [GT
Label]”]
3.Analysis:[Objectively evaluate RAG, e.g., “High-score evidence + majority consensus strongly suggest my initial
judgment may be wrong. ”]”
</RAG Reference Information Analysis>:
<Fusion, Reasoning, and Decision>:
Describe how you cross-verify RAG evidence and correct your initial visual intuition:
1. Re-examination: Re-inspect the specific areas indicated by RAG.
2. Verification (Aha! Moment): Confirm subtle artifacts missed initially.
3. Conclusion: State that initial intuition was wrong and RAG evidence successfully corrected it.
</Fusion, Reasoning, and Decision>
<Answer> Ground Truth Label </Answer>

Prompt for Resilient Rejection Chain-of-Thought
System Prompt:
You are a world-class face tampering detection analyst and an expert in critical thinking. Your task is to “role-play” and
generate aResilient Rejection Chain-of-Thought. You will receive:
1.Query Image(Primary Evidence)
2.RAG Context(Secondary Evidence: 5 retrieved reference annotations, which may bemisleadingornoisy)
3.Ground Truth Label
You must act as a rigorous expert whorejects misleading external evidence. Even if the RAG report has high similarity
scores, if it contradicts the visual facts of theQuery Image, you must identify it as “Noise” and discard it. Your
reasoning must follow three steps:
1. Visual Analysis (Independent Judgment):Analyze theQuery Imagethoroughly based on your internal forensic
knowledge. Form an initial hypothesis based strictly on visual cues (e.g., skin texture, lighting consistency).
2. RAG Analysis (Critical Scrutiny):Inspect theRAG Context. Identify evidence thatconflictswith your visual
analysis or contains hallucinations (e.g., RAG claims “blurry mouth” but the image is sharp).
3. Fusion and Decision (The “Rejection” Moment):Explicitly resolve the conflict by rejecting the noise.
1.Conflict Detection:State clearly that RAG evidence contradicts visual facts.
2.Falsification:Explainwhythe RAG evidence is invalid (e.g., “The retrieved case is a FaceSwap, but the query
image has no blending artifacts”).
3.Final Verdict:Discard the RAG evidence and stick to your internal visual judgment.
Output Format:
<Preliminary Visual Analysis>:
Describe your independent visual observation. E.g., “I observe high-frequency details in the iris and natural skin texture,
suggesting the image is likely [Real/Fake]. ”
</Preliminary Visual Analysis>
<RAG Reference Information Analysis>:
“The RAG report provides the following evidence, but it appears suspicious:”
1.Conflicting Evidence:[Cite high-scoring evidence that is factually wrong for this image]
2.Analysis:[Note the discrepancy, e.g., “Although the retriever gives a 0.85 score, the described artifacts are NOT
present in the query image. ”]
</RAG Reference Information Analysis>
<Fusion, Reasoning, and Decision>:
Perform the rejection reasoning:
1.Conflict:“RAG suggests [Wrong Label] based on [Artifact], but my visual analysis shows [Visual Fact]. ”
2.Rejection:“The retrieval is likely due to semantic similarity (e.g., similar pose) rather than forensic similarity. The
evidence is a False Positive. ”
3.Conclusion:“I reject the misleading RAG evidence and trust my internal visual analysis. ”
</Fusion, Reasoning, and Decision>
<Answer> Ground Truth Label </Answer>