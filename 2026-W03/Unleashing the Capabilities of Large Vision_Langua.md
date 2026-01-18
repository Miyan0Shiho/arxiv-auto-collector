# Unleashing the Capabilities of Large Vision-Language Models for Intelligent Perception of Roadside Infrastructure

**Authors**: Luxuan Fu, Chong Liu, Bisheng Yang, Zhen Dong

**Published**: 2026-01-15 16:16:34

**PDF URL**: [https://arxiv.org/pdf/2601.10551v1](https://arxiv.org/pdf/2601.10551v1)

## Abstract
Automated perception of urban roadside infrastructure is crucial for smart city management, yet general-purpose models often struggle to capture the necessary fine-grained attributes and domain rules. While Large Vision Language Models (VLMs) excel at open-world recognition, they often struggle to accurately interpret complex facility states in compliance with engineering standards, leading to unreliable performance in real-world applications. To address this, we propose a domain-adapted framework that transforms VLMs into specialized agents for intelligent infrastructure analysis. Our approach integrates a data-efficient fine-tuning strategy with a knowledge-grounded reasoning mechanism. Specifically, we leverage open-vocabulary fine-tuning on Grounding DINO to robustly localize diverse assets with minimal supervision, followed by LoRA-based adaptation on Qwen-VL for deep semantic attribute reasoning. To mitigate hallucinations and enforce professional compliance, we introduce a dual-modality Retrieval-Augmented Generation (RAG) module that dynamically retrieves authoritative industry standards and visual exemplars during inference. Evaluated on a comprehensive new dataset of urban roadside scenes, our framework achieves a detection performance of 58.9 mAP and an attribute recognition accuracy of 95.5%, demonstrating a robust solution for intelligent infrastructure monitoring.

## Full Text


<!-- PDF content starts -->

Unleashing the Capabilities of Large Vision-Language Models
for Intelligent Perception of Roadside Infrastructure
Luxuan Fua,1, Chong Liua,1, Bisheng Yanga,∗, Zhen Donga,b
aState Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University, Wuhan 430079, China
bHubei Luojia Laboratory, Wuhan 430079, China
Abstract
Automated perception of urban roadside infrastructure is crucial for smart city management, yet general-purpose models often
struggle to capture the necessary fine-grained attributes and domain rules. While Large Vision–Language Models (VLMs) excel
at open-world recognition, they often struggle to accurately interpret complex facility states in compliance with engineering stan-
dards, leading to unreliable performance in real-world applications. To address this, we propose a domain-adapted framework that
transforms VLMs into specialized agents for intelligent infrastructure analysis. Our approach integrates a data-efficient fine-tuning
strategy with a knowledge-grounded reasoning mechanism. Specifically, we leverage open-vocabulary fine-tuning on Ground-
ing DINO to robustly localize diverse assets with minimal supervision, followed by LoRA-based adaptation on Qwen-VL for
deep semantic attribute reasoning. To mitigate hallucinations and enforce professional compliance, we introduce a dual-modality
Retrieval-Augmented Generation (RAG) module that dynamically retrieves authoritative industry standards and visual exemplars
during inference. Evaluated on a comprehensive new dataset of urban roadside scenes, our framework achieves a detection perfor-
mance of 58.9 mAP and an attribute recognition accuracy of 95.5%, demonstrating a robust solution for intelligent infrastructure
monitoring.
Keywords:Roadside infrastructure, Attribute recognition, Multimodal vision-language model, Open-vocabulary detection.
1. Introduction
As urban development shifts from rapid expansion to refined
management, the intelligent perception of roadside infrastruc-
ture has become a critical priority [1]. This task entails not only
accurately detecting diverse facilities—such as traffic lights and
signs—but also comprehensively interpreting their fine-grained
attributes and physical conditions. However, manual inspec-
tion is prohibitively expensive, necessitating automated solu-
tions that are not only accurate but also adaptable enough to
handle this heterogeneity without imposing unmanageable an-
notation costs.
While standard Computer Vision (CV) has standardized as-
set localization, it remains bound by a rigid, data-hungry
paradigm. Traditional detectors typically rely on exhaustive su-
pervision to establish feature representations from scratch, im-
posing a heavy data burden to achieve robust performance on
domain-specific assets [2]. Moreover, without the capability
for semantic reasoning, these methods struggle to capture fine-
grained attributes—such as distinguishing specific damage con-
ditions—creating a disconnect between simple bounding boxes
and the rich, actionable insights required for intelligent mainte-
nance [3].
Recent advances in Vision–Language Models (VLMs) and
Large Language Models (LLMs) offer a promising avenue for
∗Corresponding authors.
1These authors contributed equally to this work.transcending these limitations by integrating visual recogni-
tion with natural language reasoning. These multimodal sys-
tems demonstrate exceptional capabilities in open-world per-
ception and context-aware understanding [4]. However, a criti-
cal gap remains between their descriptive prowess and the prac-
tical requirements of urban infrastructure management. Ex-
isting models are primarily tailored for general-purpose tasks
like image captioning, generating unstructured natural language
outputs that are difficult to integrate into downstream engi-
neering pipelines. Furthermore, without explicit schema-level
constraints or domain grounding, current multimodal systems
struggle to provide the structured, fine-grained state recognition
necessary for actionable real-world applications [5].
To systematically bridge this gap, this paper unleashes the
capabilities of large vision–language models in intelligent per-
ception of urban roadside scenarios and infrastructure. Specifi-
cally, we unleash the potential of representative models (e.g.,
Grounding DINO and Qwen-VL) for roadside-related detec-
tion, attribute inference, and multimodal reasoning through
fine-tuning and schema-guided structured representation. In
addition, we incorporate dual-modality knowledge integration
via a Retrieval-Augmented Generation (RAG) mechanism that
combines textual and visual retrieval. The main contributions
of this work are summarized as follows:
•Enhancing perception and reasoning via targeted fine-
tuning strategies:
Instead of relying on off-the-shelf models, we unleasharXiv:2601.10551v1  [cs.CV]  15 Jan 2026

the potential of vision-language models (VLMs) through
targeted fine-tuning strategies. Specifically, we imple-
ment open-vocabulary fine-tuning on Grounding DINO
for robust roadside infrastructure detection, and LoRA-
based fine-tuning on Qwen-VL for attribute-level reason-
ing. This design ensures precise object localization and
fine-grained interpretation of diverse roadside assets.
•Dual-modality retrieval-augmented generation:We
incorporate professional domain knowledge through a
dual-modality Retrieval-Augmented Generation (RAG)
mechanism. The proposed approach integrates pro-
fessional textual knowledge (e.g., GB 5768.2–2022
and related standards) with visual exemplars retrieved
from attribute-annotated image repositories, enabling
knowledge-grounded and structured perception for road-
side asset analysis.
•Urban roadside dataset and comprehensive evaluation:
We construct a large-scale urban roadside dataset with de-
tailed object and attribute-level annotations, covering mul-
tiple categories of infrastructure (e.g., traffic signs, lights,
bollards, hydrants, and cameras) and their diverse attribute
states. Extensive experiments conducted on datasets col-
lected from multiple cities demonstrate the effectiveness
and robustness of the proposed framework in fine-grained
attribute recognition and structured reasoning tasks.
2. Related work
Intelligent roadside perception fundamentally relies on two
core capabilities: the detection of diverse assets and the in-
terpretation of their fine-grained states. First, Sec. 2.1 traces
the shift from rigid, closed-set detectors to flexible open-
vocabulary architectures. Second, Sec. 2.2 examines method-
ologies for deciphering complex facility conditions, highlight-
ing the recent transition toward semantic reasoning powered by
Vision-Language Models.
2.1. Infrastructure Open-set Object Detection
Object detection in urban roadside scenes has long been a
cornerstone of ITS perception. Early efforts relied on classical
methods such as template matching and feature-based classi-
fiers. With the advent of deep learning, Convolutional Neu-
ral Networks (CNNs) and one-stage detectors (e.g., YOLO [6],
SSD [7]) became the standard, enabling robust detection in
complex scenes [8, 9]. While these methods achieve high accu-
racy on established benchmarks [10], they are inherently con-
strained by a closed-set paradigm. Relying on fixed categories
defined during training, they lack the flexibility to adapt to dy-
namic urban environments where novel infrastructure designs
frequently emerge.
To overcome the limitations of fixed vocabularies, Open-
V ocabulary Detection (OVD) has been developed to recog-
nize objects via arbitrary text queries. Representative ap-
proaches, such as GLIP [11] and Grounding DINO [12, 13],
reformulate detection as a phrase grounding task, leveragingpre-trained image-text embeddings (e.g., CLIP [14]) to achieve
zero-shot generalization. More recently, architectures like
YOLO-World [15] have pushed the boundaries of efficiency in
open-set scenarios. However, despite their success on general-
purpose benchmarks like COCO, these models are not specif-
ically tailored for the complexities of ITS. Their direct appli-
cability to structured roadside infrastructure remains underex-
plored, as they often lack the domain-specific granularity re-
quired for real-world maintenance, highlighting a critical need
for domain-adaptive OVD solutions.
2.2. Infrastructure Attribute State Recognition
While category-level detection provides coarse semantic la-
bels, roadside management often requires more fine-grained in-
formation regarding the attributes and physical states of infras-
tructure. For instance, determining whether a traffic light is illu-
minated or has a countdown timer [16], whether a public trash
bin is overflowing [17], or whether a traffic sign or bollard is
damaged [18, 19] has direct implications for safety and mainte-
nance. In roadside scenes, these requirements are specific and
diverse. A traffic light is not only categorized by its type but
also by its operational phases, while traffic signs must be as-
sessed for surface damage, fading, or occlusion, each conveying
critical operational information.
Existing research on such attribute-level recognition remains
sparse and disjointed. Current methodologies primarily focus
on isolated tasks, such as specific traffic light state recogni-
tion [20] or road surface damage detection [21, 22], lacking
a unified schema that covers multiple infrastructure categories.
Furthermore, traditional attribute recognition methods are often
limited by closed vocabularies and rigid label definitions, which
fail to capture the evolving nature of urban environments.
The emergence of Vision–Language Models (VLMs) of-
fers a promising direction to address these limitations by
aligning visual features with semantic understanding [23].
The field has advanced rapidly from contrastive models like
CLIP [14] to instruction-tuned architectures, such as GPT-
4V [24], LLaV A [25], and Qwen-VL [26]. Recent works like
TrafficGPT [27] have begun to explore their application within
the roadside infrastructure domain, utilizing their capabilities
in compositional understanding and Visual Question Answer-
ing (VQA) to interpret complex scenes.
However, bridging the gap between general-purpose multi-
modal agents and ITS requirements remains challenging. First,
generic VLMs often lack grounding in domain-specific stan-
dards, leading to hallucinations or failures in distinguishing
functionally similar facilities [28]. Second, while VLMs excel
at generating descriptive narratives, they predominantly pro-
duce unstructured, free-form natural language. Such outputs,
although informative, lack the structured, schema-conformant
format necessary for direct integration into automated asset
monitoring workflows [29, 30].
3. Methodology
To bridge the gap between general-purpose visual recogni-
tion and the rigorous demands of infrastructure maintenance,
2

Fig. 1.Workflow of the proposed attribute- and state-aware open-vocabulary vision-language framework.
we present a unified framework that transforms pre-trained
Vision-Language Models (VLMs) into domain-specialized
agents. As illustrated in Fig. 1, our approach operates as
a coarse-to-fine pipeline, integrating precise localization with
deep semantic reasoning. The framework consists of three syn-
ergistic components: (1) Open-V ocabulary Detection: We em-
ploy a fine-tuned Grounding DINO to accurately localize di-
verse and long-tail roadside assets with high data efficiency; (2)
Dual-Modality Knowledge Injection: To mitigate domain gaps,
a Retrieval-Augmented Generation (RAG) mechanism dynam-
ically retrieves authoritative industry standards and visual ex-
emplars as contextual prompts; and (3) Attribute-Aware Rea-
soning: A LoRA-adapted Qwen-VL synthesizes the visual fea-
tures and retrieved knowledge to infer fine-grained attributes.
The following subsections detail the multimodal framework de-
sign (Sec. 3.1), the object detection strategy (Sec. 3.2), and the
knowledge-enhanced attribute reasoning process (Sec. 3.3).
3.1. Multimodal Detection-to-Attribute Framework
The framework integrates the fine-tuned Grounding DINO
and Qwen-VL models into a unified multimodal perception and
dialogue system. As shown in Fig. 1, the end-to-end process
spans open-vocabulary detection to interactive reasoning.
Starting with panoramic images of urban road scenes, the
images are divided into four perspective sub-images to improve
detection granularity. The Grounding DINO model, fine-tuned
for open-vocabulary detection, extracts category labels, con-
fidence scores, and bounding boxes. These boxes are then
cropped and processed by the fine-tuned Qwen-VL model,
trained with LoRA on the attribute annotation dataset from Sec-
tion 4.1. Guided by specific prompts, Qwen-VL performs at-
tribute reasoning and generates structured descriptions.
To generate standardized, machine-readable outputs from
this vision–language pipeline, we employ a structured JSON
representation for roadside scene analysis. Unlike unstructured
captions, the JSON format enforces a consistent schema across
categories and attributes, ensuring interoperability and support-
ing large-scale quantitative evaluation. As illustrated in Fig. 2,the final JSON output includes each object’s category, bounding
box, and attribute–confidence pairs. This unified structure sup-
ports both single-image inference and large-scale dataset pro-
cessing, enabling seamless integration with infrastructure per-
ception and management systems.
Fig. 2.Example of structured JSON representation for roadside object
attributes.
The transformation from detection to JSON involves two
stages: (i) extracting bounding boxes and class labels from
Grounding DINO as object instances, and (ii) feeding cropped
regions with contextual prompts into Qwen-VL to infer fine-
grained attributes according to the predefined schema. For sim-
3

ple attributes (e.g., object posture or light state), rule-based cues
are embedded within the prompts to ensure stable inference.
This two-stage process bridges perception and structured rea-
soning, allowing the system to output both interpretable and
machine-readable results.
Additionally, the fine-tuned Qwen-VL enables multimodal
dialogue, allowing users to query specific attributes, states,
or conditions of roadside infrastructure directly from the im-
ages. The system integrates detection results, structured anno-
tations, and panoramic visualizations into an interactive frame-
work for intelligent urban roadside management. This uni-
fied system links visual grounding, attribute reasoning, and
human–machine interaction, supporting both automated struc-
tured perception and interactive querying.
3.2. Roadside Infrastructure Object Detection
To tailor vision-language models (VLMs) such as
Grounding-DINO to the specific requirements of urban
roadside scene understanding, we design and evaluate three
fine-tuning strategies with different levels of semantic flexibil-
ity: closed-set fine-tuning, open-set continued pre-training, and
open-vocabulary fine-tuning (OVF). Each strategy employs a
distinct model architecture configuration and parameter update
policy, reflecting the trade-offbetween specialization and
generalization.The model architecture diagrams of these three
fine-tuning strategies are illustrated in Fig. 3.
Closed-set fine-tuning:Closed-set fine-tuning follows the
conventional paradigm in which the model is trained only on
categories available in the annotated dataset. The detection
head is aligned with the target classes, while the visual back-
bone and language encoder remain frozen to retain pre-trained
semantics. This achieves high accuracy for known roadside ob-
jects such as traffic signs or traffic lights but fails to recognize
novel or evolving facilities.
Open-set continued pre-training fine-tuning:To improve
adaptability while preserving learned knowledge, open-set con-
tinued pre-training selectively updates the visual backbone,
keeping the language encoder fixed. This allows gradual re-
finement of visual features and adaptation to new environments
while maintaining semantic grounding. Negative sampling is
employed to distinguish roadside-relevant regions from back-
ground clutter, mitigating hallucination. As a result, this strat-
egy enhances robustness and domain generalization in complex
urban scenes.
Open-vocabulary fine-tuning:Building on this, open-
vocabulary fine-tuning (OVF) serves as the core of our ap-
proach. Rather than relying on fixed class labels, OVF con-
ditions detection on natural language prompts, enabling the
model to link unseen visual patterns with descriptive queries.
The dataset is divided into Base classes (e.g., vehicles, pedestri-
ans) and Novel classes (e.g., traffic lights, bollards). The model
is trained only on Novel classes and evaluated jointly on both,
testing its generalization capability. OVF jointly updates the
backbone and language encoder with controlled learning rates
to mitigate catastrophic forgetting. This design maintains open-
domain knowledge while enhancing generalization to unseen
roadside categories via descriptive prompts.In summary, while closed-set fine-tuning provides strong
accuracy for fixed categories and open-set pre-training im-
proves robustness, open-vocabulary fine-tuning uniquely main-
tains prior knowledge while enabling scalable detection of un-
seen objects—making it the most effective strategy for real-
world intelligent roadside systems where infrastructure evolves
continuously.
Fig. 3.Model architectures of the three fine-tuning strategies.
3.3. Roadside Infrastructure Attribute State Recognition
To enable precise fine-grained attribute recognition for road-
side infrastructure, we introduce a specialized module that
adapts Vision-Language Models into domain-expert agents. As
shown in Fig. 4, it integrates attribute-guided fine-tuning with
dual-modality Retrieval-Augmented Generation (RAG). The
module comprises two components: (1) Attribute-Guided Fine-
Tuning: LoRA adaptation of Qwen-VL aligned to structured
attribute schemas via visual instruction tuning; and (2) Dual-
Modality RAG: Retrieval of textual standards and visual exem-
plars to ground attribute inference in domain knowledge. The
following subsections describe the fine-tuning strategy (Sec.
3.3.1) and the RAG-enhanced reasoning (Sec. 3.3.2).
3.3.1. Attribute-Guided Fine-tuning for VLMs
Although pre-trained vision–language models possess strong
general perception capabilities, they often lack the domain-
specific granularity required to accurately interpret complex
roadside assets and struggle to strictly adhere to predefined at-
tribute schemas [31]. To address these limitations and adapt the
general-purpose model into a domain specialist, we implement
an attribute-guided fine-tuning strategy.
Building upon the structured attribute annotations, we fine-
tune Qwen-VL to improve attribute reasoning and domain
adaptation performance. The core objective is to enable the
model to parse roadside infrastructure states with high preci-
sion while maintaining rigorous output formatting. To achieve
4

Fig. 4.Overview of the dual Retrieval-Augmented Generation (RAG) framework, integrating Textual RAG for knowledge-grounded reasoning
using domain standards, and Visual RAG for exemplar-based attribute retrieval from annotated image databases.
this, we formulate the attribute recognition task as a supervised
visual instruction tuning problem [12].
The annotated data described in Section 4.1 is trans-
formed into instruction-following pairs, structured as
⟨Image,Instruction,Output⟩. For each training instance,
the input consists of the cropped object image and a prompt
template (e.g.,“Identify the attributes of the roadside object
in the image following the standard schema. ”), while the
target output is the ground-truth JSON sequence. This format
forces the model to learn the correspondence between visual
features and the specific attribute schema defined for urban
infrastructure.
Fine-tuning is conducted using parameter-efficient Low-
Rank Adaptation (LoRA) [32], which minimizes computational
overhead while preserving the model’s generalization capabil-
ity. Instead of updating the entire parameter space of the Large
Language Model (LLM), we freeze the pre-trained vision en-
coder and the majority of the LLM backbone. LoRA adapters
are injected specifically into the projection layers (e.g., query,
key, value projections) of the attention mechanisms within the
transformer blocks.
Mathematically, for a pre-trained weight matrixW, LoRA
introduces a low-rank decomposition structure:
W′=W+ ∆W=W+AB(1)
whereAandBare low-rank matrices, and∆Wrepresents
the update learned during fine-tuning. This structure al-
lows the model to adapt efficiently to domain-specific knowl-
edge—such as distinguishing specific sign types or damage
conditions—without suffering from catastrophic forgetting of
its foundational capabilities.
During training, the model is supervised using the structured
JSON sequences. The optimization objective is to maximize
the probability of generating the correct attribute tokens andstructure markers given the visual input and textual instruc-
tions. This focused supervision enables the model to transition
from generating free-form descriptions to producing rigorous,
schema-conformant outputs suitable for automated engineering
downstream tasks.
After fine-tuning, Qwen-VL exhibits enhanced consistency
between perception and reasoning modules. When integrated
with the fine-tuned Grounding DINO, the system demonstrates
improved attribute recognition accuracy, reduced hallucination,
and stronger robustness under diverse urban conditions.
3.3.2. Domain Knowledge Imbedding via RAG
To compensate for the dual gap in domain-specific knowl-
edge and exemplar-based visual experience that limits reliable
attribute inference, we incorporate a dual-modality Retrieval-
Augmented Generation (RAG) mechanism that integrates both
professional textual knowledge and visual exemplars into the
multimodal reasoning process. RAG [33] combines external in-
formation retrieval with generative modeling, allowing large vi-
sion–language models to access structured knowledge and his-
torical visual experience during inference.
As illustrated in Fig. 4, the proposed framework includes two
complementary RAG branches: (1) atextual RAGfor integrat-
ing domain-specific documents, and (2) avisual RAGfor lever-
aging image-based retrieval to enhance object-level attribute
reasoning.
Textual RAG for Knowledge-Grounded Reasoning:In
the textual RAG branch, adomain knowledge baseis con-
structed by parsing, segmenting, and encoding authoritative ref-
erences such as roadside regulations, signage standards, and
infrastructure manuals. National and regional standards (e.g.,
GB 5768.2–2022Road Traffic Signsand related specifications)
are transformed into structured vector embeddings and stored
in a local semantic database for retrieval [34].
Given a user queryq textrelated to attribute inference (e.g.,“Is
5

this a warning sign?”), the system performs semantic matching
to retrieve the top-kmost relevant text fragments{d 1,d2, . . . ,d k}
from the knowledge baseD. The retrieval process is formulated
as:
{d1, . . . ,d k}=Top-kn
sim(e q,edi)|d i∈Do
,(2)
wheree qande didenote the text embeddings of the query and
document fragmentd i, respectively, and sim(·,·) represents the
cosine similarity:
sim(e q,edi)=eq·edi
∥eq∥∥edi∥.(3)
These retrieved definitions and rules are dynamically ap-
pended to the model’s contextual input, enabling the multi-
modal model (Qwen-VL) to performknowledge-grounded rea-
soning. This process ensures that predictions adhere to formal
domain conventions, such as recognizing that “warning signs”
typically adoptyellow triangularconfigurations, as defined in
GB 5768.2–2022 and related standards.
Visual RAG for Object-Level Attribute Retrieval:In
addition to textual grounding, we design a visual retrieval-
augmented generation pipeline that enhances object-level at-
tribute recognition through exemplar-based reasoning [35]. For
each detected roadside object, the system first encodes its
cropped imageI cropinto a visual embeddingv query using a pre-
trained CLIP [14] vision encoder:
vquery=f CLIP(Icrop),(4)
wheref CLIP(·) represents the CLIP image encoder that maps the
cropped image into a normalized feature space.
Subsequently, a similarity search is performed within an
attribute-annotated image databaseV={(I i,ai)}N
i=1, whereI i
denotes thei-th reference image anda irepresents its corre-
sponding attribute annotations. The retrieval identifies the top-
mmost visually similar samples based on cosine similarity:
{(I∗
1,a∗
1), . . . ,(I∗
m,a∗
m)}=Top-mn
sim(v query ,vi)|(I i,ai)∈Vo
,
(5)
wherev i=f CLIP(Ii) denotes the pre-computed embedding of
reference imageI i, and the cosine similarity is computed as:
sim(v query ,vi)=vquery·vi
∥vquery∥∥vi∥.(6)
The retrieved attribute annotations{a∗
1, . . . ,a∗
m}are then in-
jected as contextual inputs into the multimodal model. This
process enables Qwen-VL to reference prior visual–semantic
correspondences, leveraging historical annotation experience to
produce more accurate and consistent attribute predictions. For
example, when identifying a "spherical bollard with reflective
strips," the retrieved exemplars provide prior knowledge about
surface reflectivity or damage patterns, improving fine-grained
attribute reasoning.
Unified Knowledge-Augmented Generation:By integrating both textual and visual retrieval, the system
transitions from perceptual pattern recognition to knowledge-
grounded, context-enriched reasoning. A concrete inference
example is presented in Fig. 5, where the retrieval of the stan-
dard definition (e.g., circular shape, blue background) and visu-
ally similar exemplars effectively guides the model to generate
precise, schema-conformant attributes for a mandatory traffic
sign. The final input to the vision-language model combines
the original image, the retrieved textual knowledge{d 1, . . . ,d k},
and the visual exemplar annotations{a∗
1, . . . ,a∗
m}, formulated as:
Output=VLM(I crop ,Ioriginal ,{d1, . . . ,d k},{a∗
1, . . . ,a∗
m}),(7)
where VLM(·) denotes the vision-language model that performs
multimodal reasoning over the augmented context.
The textual RAG provides formal semantic constraints de-
rived from standards, while the visual RAG supplies perceptual
analogs drawn from prior samples. Together, they enhance in-
terpretability, reduce hallucination, and ensure that generated
attributes are both visually grounded and semantically compli-
ant with professional definitions. Moreover, new documents or
annotated samples can be continuously added to the knowledge
base without retraining, ensuring scalability and adaptability
for evolving roadside infrastructures.
Fig. 5.A case study of the RAG-enhanced inference process. The sys-
tem retrieves the “Mandatory sign” definition from GB 5768.2–2022
and visually similar exemplars to guide the generation of structured
attribute outputs for a blue circular traffic sign.
4. Experiments and Results
This section presents a comprehensive evaluation of the pro-
posed framework. We begin by introducing the annotated
6

Fig. 6.Examples of object detection annotations from the roadside infrastructure dataset in urban roadside scenes.
Fig. 7.The distribution of annotation numbers among different se-
mantic categories in our dataset. The blue and yellow bars represent
the statistics for the Shanghai and Wuhan subsets, respectively.
dataset and evaluation metrics, followed by an assessment
of the system’s capabilities in open-vocabulary detection and
fine-grained attribute recognition. Finally, we conduct exten-
sive comparative experiments against representative baselines
to validate the effectiveness of the proposed fine-tuning and
retrieval-augmented strategies.
4.1. Attribute-Annotated Roadside Infrastructure Dataset
To facilitate fine-grained perception and multimodal reason-
ing within complex urban environments, we construct a large-
scaleurban roadside infrastructure datasetenriched with
object-level annotations and fine-grained attribute labels. The
dataset serves as a comprehensive benchmark for both percep-
tion and attribute reasoning tasks within complex urban scenes,
advancing beyond traditional benchmarks that focus primarily
on segmentation or coarse detection [36, 37].
The dataset is constructed from panoramic roadside images
captured by vehicle-mounted sensors across two major Chi-
nese cities: Shanghai and Wuhan. In total, the dataset contains3,551 high-resolution panoramic images (8192×4096), com-
prising 1,576 images from Shanghai and 1,975 from Wuhan.
For the Shanghai subset, we use 1,316 images for training and
260 images for validation. For the Wuhan subset, the training
split includes 1,746 images, while the validation split contains
229 images. This city-specific partitioning supports the subse-
quent experiments, where each city is evaluated independently
as well as under cross-city generalization settings.
This dataset includes more than 100,000 annotated instances
across ten representative infrastructure categories: traffic signs,
signal lights, street lights, surveillance cameras, bollards, ball
bollards, fire hydrants, trash bins, manhole covers, and traffic
cones. The quantitative distribution of annotations across these
categories for both Shanghai and Wuhan is illustrated in Fig. 7.
These annotations provide rich object-level and attribute-level
information, supporting a variety of perception and reasoning
tasks. Examples of the object detection annotations across these
categories are shown in Fig. 6 and Fig. 8, illustrating the anno-
tated objects in urban roadside scenes.
To support fine-grained and interpretable annotation of road-
side infrastructure, we construct a structured attribute schema
that organizes common urban roadside objects and their asso-
ciated properties. As detailed in Table 1, the dataset covers ten
representative categories, where each category is assigned a set
of practical semantic attributes such as shape, color, material,
operational status, and physical condition.
Unlike conventional datasets that provide only object labels,
our schema offers clear definitions for each attribute and its pos-
sible values. This helps ensure annotation consistency and pro-
vides a detailed reference for subsequent model training and
evaluation. During annotation, the schema serves as a guideline
for describing what information should be recorded for each
object instance, enabling the dataset to capture not only object
categories but also their states and functional characteristics.
7

Fig. 8.Examples from the roadside infrastructure dataset, illustrating diverse categories, illumination conditions, and attribute variations.
By establishing this attribute-centric structure, we bridge the
gap between low-level perception and high-level semantic rea-
soning, thereby improving interpretability and supporting struc-
tured interaction in downstream roadside analysis systems.
Table 1
Attribute schema for ten categories of urban roadside facilities.
Category Attributes
Traffic Light Type, Working State, Color, Damage Condition,
Device Type ...
Street Light Number of Arms, Working State, Damage Condi-
tion, Solar-Powered ...
Traffic Sign Type, Shape, Color, Damage Condition ...
Bollard Material, Color, Posture, Damage Condition, Re-
flective Property ...
Ball Bollard Category, Posture, Damage Condition, Reflective
Property ...
Surveillance Camera Occlusion Condition, Damage Condition, Shape,
Application scenario ...
Manhole Cover Shape, Safety Condition, Surface Pattern ...
Trash Bin Category, Material, Color, Shape, Fullness, Dam-
age Condition, Lid Condition, Nearby Garbage
Piles, Fixed Type ...
Fire Hydrant Color, Working State, Damage Condition ...
Traffic Cone Color, Posture, Damage Condition, Reflective
Property ...4.2. Performance of Roadside Infrastructure Detection
In the open-set setting, open-vocabulary detectors transcend
the limitation of recognizing only classes predefined in a fixed
training set. This capability is fundamental for building a com-
prehensive inventory of urban roadside infrastructure, where
object categories can be diverse and evolving. In this ex-
periment, we evaluate the open-vocabulary detection perfor-
mance in complex roadside scenes. As shown in Fig. 9(a),
the model effectively leverages language-guided grounding to
identify standard roadside-related objects such as traffic signs,
signal lights, and bollards within cluttered urban environments.
Beyond these common entities, the detector also generalizes to
a broader range of facilities, including surveillance cameras,
barriers, fire hydrants, trash bins, and manhole covers. This
demonstrates the flexibility of language-guided grounding in
handling diverse infrastructure categories based on semantic
descriptions rather than rigid classes.
Furthermore, leveraging its Referring Expression Compre-
hension (REC) capabilities, the model can identify objects
based on specific attributes or conditions, which is crucial for
dynamic scene analysis. As illustrated in the detailed views
of Fig. 9(b), for roadside infrastructure, the model successfully
identifies traffic lights and differentiates their operational states
such as red, green, and yellow. It also detects abnormal condi-
tions including not-working or falling lights. In terms of traf-
fic sign perception, the model distinguishes prohibitory, warn-
ing, and mandatory signs, and further detects defects such as
faded or damaged signs. These abilities are important for facil-
ity maintenance and roadside regulation management.
While this detection-level attribute awareness provides im-
8

mediate visual alerts for maintenance, it is primarily unstruc-
tured and lacks professional domain knowledge. To achieve
standardized and machine-readable asset management, these
detection results serve as the pivotal input for the subsequent
multimodal reasoning module, which generates the structured
JSON outputs detailed in the following sections.
To strictly quantify the reliability of these detections before
they are utilized for downstream reasoning, we adopt standard
object detection metrics commonly used in the literature.
For detection accuracy, we first employ Intersection over
Union (IoU), which quantifies the spatial overlap between a
predicted bounding box and its corresponding ground-truth an-
notation:
IoU=Area o f Union
Area o f Overlap(8)
IoU ranges from 0 to 1, where larger values indicate more ac-
curate localization. In practice, detections are considered cor-
rect when IoU exceeds predefined thresholds (e.g., 0.5 or 0.75),
reflecting different strictness levels of spatial alignment.
Based on IoU-matched predictions, we further report Preci-
sion and Recall to characterize detection reliability and com-
pleteness:
Precision=T P
T P+FP
Recall=T P
T P+FN(9)
Precision evaluates the proportion of correct predictions among
all detected objects, while Recall measures the proportion of
ground-truth objects that are successfully detected. Together,
they reflect the trade-offbetween false alarms and missed de-
tections in real-world roadside scenes.
To provide a unified measure over different recall levels, we
adopt Average Precision (AP), which summarizes the preci-
sion–recall curve as:
AP=Z1
0P(r)dr, (10)
whereP(r) denotes the precision at recall levelr. In this work,
AP follows the definition and numerical integration protocol es-
tablished by the COCO benchmark, and detailed computation
procedures can be found in the original COCO evaluation pa-
per [38].
Finally, Mean Average Precision (mAP) is reported as the
mean AP across all evaluated categories and serves as the pri-
mary indicator of overall detection performance. We adopt
commonly used variants, including mAP@50 and mAP@75,
which correspond to IoU thresholds of 0.5 and 0.75, respec-
tively, as well as mAP@50:95, which averages AP over IoU
thresholds from 0.5 to 0.95 with a step size of 0.05. This multi-
threshold evaluation provides a comprehensive assessment of
detection robustness under varying localization requirements.
Applying these metrics to our collected data, we first ana-
lyze the fine-grained performance across different infrastruc-
ture types. Table 2 details the per-class detection accuracy on
Fig. 9.Open-vocabulary perception and fine-grained state analysis of
roadside infrastructure. (a) Open-set inventory of diverse infrastruc-
ture categories. (b) REC-based attribute reasoning, capturing opera-
tional states (e.g., signal colors, malfunctions), physical defects (e.g.,
damage, fading), and semantic interpretations (e.g., sign types and di-
rectionality).
the Shanghai and Wuhan datasets after fine-tuning. It is ob-
served that while most categories achieve satisfactory results,
certain classes like surveillance cameras and manholes show
relatively lower mAP. This performance gap is primarily due
to their smaller physical scale and the challenge of identifying
subtle visual features from fixed roadside perspectives.
9

Fig. 10.Example of Attribute Recognition in Roadside Infrastructure.
Table 2
Per-class detection performance of roadside infrastructure categories
on the Shanghai and Wuhan datasets after fine-tuning.
CategoryShanghai Wuhan
mAP mAP@50 mAP mAP@50
Traffic light 68.3 87.7 67.2 89.7
Fire hydrant 56.0 92.7 51.4 80.9
Street light 61.5 79.9 64.2 81.9
Traffic sign 63.9 85.5 63.8 77.7
Bollard 49.4 89.7 62.1 90.7
Surveillance camera 30.2 65.3 39.7 75.9
Manhole 49.5 70.2 44.2 68.7
Trash bin 49.5 68.8 73.8 91.9
Ball bollard 58.7 78.7 60.0 80.8
Traffic cone 49.8 84.6 62.5 87.0
All 53.2 80.3 58.9 82.5
We further evaluate the robustness and generalization ca-
pability of the proposed detection framework across different
urban environments. Specifically, experiments are conducted
on two large-scale roadside datasets collected in Wuhan and
Shanghai, which follow the same annotation protocol and eval-
uation metrics. Two evaluation settings are considered: in-
domain evaluation, where training and testing are performed
within the same city, and cross-city evaluation, where a model
trained on one city is directly applied to the other without ad-
ditional adaptation. This setting reflects realistic deployment
scenarios in which roadside perception systems are expected to
operate reliably in previously unseen cities.
Table 3 summarizes the multi-city detection performance ofthe Open-V ocabulary Fine-tuning model with a Swin-T back-
bone. The diagonal entries correspond to in-domain results,
where strong detection accuracy is achieved in both cities.
Among them, Wuhan exhibits the highest in-domain perfor-
mance, reaching an mAP of 0.589 and an mAP@50 of 0.825.
Such variations can be attributed to differences in scene com-
plexity, infrastructure density, and visual appearance across ur-
ban environments.
Despite noticeable differences in roadside layouts and object
characteristics between the two cities, the cross-city evaluation
results (off-diagonal entries in Table 3) demonstrate that models
trained on one city maintain reasonable detection performance
when transferred to the other. In most cases, the performance
degradation remains moderate compared with in-domain evalu-
ation, indicating that the detector captures transferable semantic
and structural patterns rather than relying on city-specific visual
cues. Moreover, the generalization behavior is relatively sym-
metric across the two cities, with Shanghai showing slightly
stronger mutual transferability. Overall, these results indicate
that the proposed detection framework exhibits solid robust-
ness under domain shifts and is suitable for large-scale urban
deployment scenarios.
4.3. Performance of Roadside Infrastructure Attribute Recog-
nition
Beyond category detection, an essential feature of our frame-
work is its ability to capture fine-grained attributes of roadside
infrastructure. Rather than merely recognizing the object class,
the system provides structured information on operational state,
material, condition, shape, and color, enabling a more compre-
hensive understanding of urban roadside environments.
As illustrated in Fig. 10, multiple categories of facilities
10

Table 3
Multi-city evaluation: in-domain accuracy and cross-city generaliza-
tion.
Train City Test City Setting mAP mAP@50
WuhanWuhan In-domain 0.589 0.825
Shanghai Cross-city 0.406 0.651
ShanghaiShanghai In-domain 0.532 0.803
Wuhan Cross-city 0.469 0.695
are simultaneously detected and annotated with their attributes
within the same scene. For traffic lights, the system distin-
guishes light colors (red, green, or yellow) and detects abnor-
mal conditions such as malfunctioning bulbs or tilted poles.
For traffic signs, it recognizes both prohibitory and mandatory
types, further describing geometric shape (circular), dominant
colors (red, blue, white), and surface integrity. Street lamps are
annotated with structural and functional properties such as the
number of lamp arms, illumination status, and damage condi-
tion. Ancillary facilities such as trash bins are characterized by
waste type (recyclable or other waste), material (metal), color
scheme, lid status, and nearby garbage accumulation.
By jointly presenting object categories and attribute meta-
data in a structured format, the system extends perception from
simple recognition to detailed condition analysis. This allows
roadside management systems to assess infrastructure status,
identify damaged or malfunctioning assets, and prioritize main-
tenance operations. The combination of accurate detection
and fine-grained attribute reasoning thus ensures a more inter-
pretable, actionable, and safety-oriented perception framework
for intelligent urban roadside management.
Structured JSON Answering: the system supports a struc-
tured answering mode based on the Attribute-Based Schema.
In this mode, each detected object is output in JSON format,
including its category, bounding box coordinates, fine-grained
attributes (e.g., color, condition, operational status), and confi-
dence score. This structured output enables machine-readable
results for downstream integration, while remaining consistent
with the visual evidence.
Beyond emitting schema-aligned JSON outputs, the fine-
tuned model also supports open-ended multimodal dialogue.
Figure 11 shows a typical query–image pair in which the user
asks about the textual and directional guidance visible on the
overhead road board. The model reads and returns the salient
content, e.g.,“Middle Ring Rd. ”and the guidance“Right To
Shangzhong Rd. Tunnel; Left To Pudong Airport, ”together
with the indicated turning directions shown by the arrows. This
example illustrates that semantics such as place names and
routing instructions—information not naturally captured by at-
tribute schemas—can be obtained through dialogue while main-
taining image grounding. In practice, LoRA-based adaptation
and domain grounding yield concise, accurate answers without
hallucinated attributes, demonstrating that the system handles
free-form queries alongside structured predictions.
While these qualitative results illustrate the capability of gen-
erating standardized records, ensuring the reliability of these
Fig. 11.Qualitative example of open-ended dialogue after fine-tuning:
the model answers questions about road guidance content in the scene.
data for automated maintenance requires quantitative verifica-
tion. To rigorously evaluate the correctness of these structured
JSON outputs for multimodal attribute reasoning, we define a
specializedAttribute Accuracymetric. This metric calculates
the ratio of correctly predicted attributes to the total number of
attributes across all objects:
Accuracy=PM
j=1PKj
k=1I(aG
jk=aP
jk)
PM
j=1Kj(11)
whereMdenotes the number of objects,K jis the number of
attributes for objectj,aG
jkandaP
jkare the ground-truth and pre-
dicted values of attributekfor objectj, andI(·) is the indicator
function. This metric directly evaluates how well the gener-
ated attributes match the ground-truth annotations, providing a
quantitative basis for the system’s reasoning performance.
Guided by this metric, we performed a detailed quantitative
assessment to verify the model’s precision across different facil-
ity types. To further analyze attribute recognition performance
at a finer granularity, Table 4 reports the per-class attribute ac-
curacy of roadside infrastructure categories on the Shanghai
and Wuhan datasets. The results indicate that categories like
traffic lights and signs exhibit slightly lower accuracy due to
the complexity of distinguishing fine-grained sub-types, such
as motor vehicle versus pedestrian signals. Moreover, small-
scale objects like surveillance cameras and ball bollards present
greater challenges for attribute recognition, while the perfor-
mance fluctuations between Shanghai and Wuhan reflect vari-
ations in scene layouts and the diverse visual appearances of
infrastructure across different urban environments.
11

Table 4
Per-class attribute recognition performance of roadside infrastructure
categories on the Shanghai and Wuhan datasets after fine-tuning and
dual-modality RAG.
Category Shanghai Accuracy Wuhan Accuracy
Traffic light 91.6 85.8
Fire hydrant 96.2 97.9
Street light 99.0 97.7
Traffic sign 90.1 95.1
Bollard 98.9 98.9
Surveillance camera 91.1 91.4
Manhole 97.7 97.1
Trash bin 97.4 97.2
Ball bollard 80.7 89.5
Traffic cone 97.1 99.6
All 95.5 94.1
4.4. Comparative Experiments
4.4.1. Object Detection Comparison
(1) Effectiveness of Fine-Tuning (Before vs. After):
Accurate detection of roadside infrastructure serves as the
foundation of the proposed framework. Fig. 12 presents a
comparative visualization of detection results before and af-
ter fine-tuning. Fig. 12(b) shows the outputs of the fine-tuned
model, while Fig. 12(a) depicts the baseline results. It can
be clearly observed that after domain-specific fine-tuning, the
model achieves more precise and complete detection, success-
fully identifying small, slender, and complex infrastructure el-
ements that were previously missed or misclassified, such as
streetlight poles, fire hydrants, and traffic cones. The bounding
boxes are more stable and better aligned with object boundaries,
reflecting enhanced localization robustness and higher category
confidence. These improvements verify the effectiveness of the
open-vocabulary fine-tuning strategy and confirm that the ten
representative classes of roadside infrastructure can now be re-
liably and consistently recognized.
Fig. 12.Comparison of the after fine-tuning and before fine-tuning
detection results
To validate these qualitative observations with quantitativeevidence, we use the Wuhan dataset as a representative ex-
ample to illustrate the effects of different fine-tuning strate-
gies. As shown in Table 5, the quantitative results reveal that
fine-tuning greatly enhances the detection ability of novel cat-
egories. Specifically, the mAP@50 of traffic lights improves
from 57.9 to 89.7, bollard from 1.8 to 90.7, and ball bollards
from 0.0 to 80.8. These improvements demonstrate the model’s
capability to capture discriminative features of small-scale and
underrepresented objects after fine-tuning.
However, certain categories remain challenging. Surveil-
lance cameras, for instance, improve from 8.2 to 75.9 in
mAP@50, but this is still lower than other categories. The pri-
mary reasons include limited training samples and small object
sizes, which hinder robust detection. Nevertheless, the fine-
tuned model consistently outperforms the baseline across al-
most all novel categories.
Overall, these improvements verify the effectiveness of the
open-vocabulary fine-tuning strategy and confirm that the ten
representative classes of roadside infrastructure can now be re-
liably and consistently recognized.
Table 5
Novel class detection accuracy comparison on Wuhan dataset (before
vs. after fine-tuning).
CategoryZero-shot Fine-tuning
mAP mAP@50 mAP mAP@50
Traffic light 33.3 57.9 67.2 89.7
Fire hydrant 34.8 61.3 51.4 80.9
Street light 30.2 40.7 64.2 81.9
Traffic sign 28.4 38.9 63.8 77.7
Bollard 0.7 1.8 62.1 90.7
Surveillance camera 3.3 8.2 39.7 75.9
Manhole 7.7 12.6 44.2 68.7
Trash bin 20.9 30.5 73.8 91.9
Ball bollard 0.0 0.0 60.0 80.8
Traffic cone 21.1 37.1 62.5 87.0
All 18.0 28.9 58.9 82.5
(2) Comparison with Representative Detection Models:
We further compare MM-Grounding DINO (MM-GD)
with representative open-vocabulary detectors (GLIP,
YOLO-World [15], and OV-DINO [39]) on the roadside
detection benchmark under both zero-shot and fine-tuning set-
tings. GLIP, OV-DINO, and MM-GD use the same backbone
(Swin-T) and the same pretraining corpora (Objects365 [40]
and GoldG), while YOLO-World employs YOLOv8_L with
Objects365 [40], GoldG, and CC-LiteV2 (a newly annotated
250k subset of CC3M [41]). As evidenced in Table 6, MM-GD
achieves the best zero-shot performance among the four models
(Roadside mAP and mAR of 18.0 and 28.9, respectively),
with clear gains in Roadside mAP@50 over GLIP (+13.5),
YOLO-World (+6.0), and OV-DINO (+2.7).
After closed-set fine-tuning on the roadside dataset, MM-GD
further widens the margin. In particular, Roadside mAP@50
reaches 83.1, surpassing OV-DINO (+4.3), YOLO-World
(+14.0), and GLIP (+35.2). MM-GD also leads on Roadside
mAP (59.2) and Roadside mAR (60.6). These results confirm
12

Table 6
Architecture comparison on the roadside benchmark under zero-shot and closed-set fine-tuning settings on Wuhan dataset.
Architecture Pre-Train Data BackboneRoadside
mAP@50Roadside
mAPRoadside
mAR
YOLOv11 (from-scratch) - YOLOv11-n 24.4 12.2 34.2
GLIP (zero-shot) O365, GoldG Swin-T 15.4 10.4 16.2
YOLO-World (zero-shot) O365+GoldG+CC-LiteV2 YOLOv8_L 22.9 13.9 23.7
OV-DINO (zero-shot) O365, GoldG Swin-T 26.2 16.7 28.3
MM-GD (zero-shot) O365, GoldG Swin-T 28.9 18.0 28.4
GLIP (fine-tuning) O365, GoldG Swin-T 47.9 28.1 32.8
YOLO-World (fine-tuning) O365+GoldG+CC-LiteV2 YOLOv8_L 69.1 41.7 49.3
OV-DINO (fine-tuning) O365, GoldG Swin-T 78.8 51.4 57.8
MM-GD (fine-tuning) O365, GoldG Swin-T83.1 59.2 60.6
that MM-GD delivers the strongest performance both in the
zero-shot regime and after fine-tuning, especially on the key
metrics Roadside mAP@50 and Roadside mAR.
To further contextualize these results, we compare from-
scratch training on a recent YOLOv11 baseline against fine-
tuned MM-GD on the same dataset. As shown in Table 6,
the from-scratch YOLOv11 model underperforms dramati-
cally across Roadside mAP@50, Roadside mAP, and Road-
side mAR, reflecting severe overfitting due to limited train-
ing data and weak generalization. In contrast, the fine-tuned
MM-GD (Swin-T/Swin-B) attains substantially higher accu-
racy across all metrics. This comparison highlights that—in
data-constrained roadside scenarios—adapting a strong object
detector via fine-tuning is markedly more effective than train-
ing from scratch.
(3) Comparative Analysis of Fine-Tuning Strategies:
Finally, we evaluate different fine-tuning strategies in Ta-
ble 7: closed-set fine-tuning substantially boosts performance
on predefined roadside categories relative to zero-shot base-
lines. However, the last column also reveals a severe limitation:
COCO mAP collapses to near zero after closed-set fine-tuning,
indicating poor generalization outside the roadside-specific la-
bel space.
Moving beyond closed-set training, Table 7 shows that open-
set continued pretraining yields a more balanced outcome:
roadside mAP drops slightly relative to closed-set results, but
COCO mAP improves noticeably. This suggests better trans-
fer to generic distributions while maintaining reasonable in-
domain performance.
The open-vocabulary fine-tuning results in Table 7 demon-
strate clear gains on novel roadside categories while constrain-
ing losses on base coco classes. With base/novel splits and
domain prompts, the model substantially improves novel road-
side detection, while base coco performance remains relatively
stable. The improvement persists under a larger backbone, al-
beit with modest increments. Moreover, the box mAP (IoU
0.50:0.95) indicates the system retains fine localization capa-
bility on both novel and base classes, though certain base cate-
gories still leave room for improvement.
It is also noteworthy that while novel categories achieve sig-
nificant accuracy gains, some base categories show slight de-Table 7
Summary across fine-tuning regimes under identical evaluation on
Wuhan dataset.
Fine-tuning mode Backbone Roadside mAP COCO mAP
Zero-shotSwin-T 18.0 50.4
Swin-B 18.5 52.5
Closed-setSwin-T 57.2 0.1
Swin-B 57.7 0.2
Open-set continued pretrainSwin-T 45.7 7.2
Swin-B 46.4 8.3
Open vocabularySwin-T 58.9 47.4
Swin-B 59.0 47.6
creases in performance. This trade-offis expected, as the fine-
tuning process focuses on learning new categories, which may
slightly impact the generalization ability for previously known
categories. The design goal of open-vocabulary fine-tuning is
to strike a balance: maximizing precision for new categories
while maintaining reasonable performance on base categories.
In conclusion, closed-set training maximizes roadside accu-
racy but sacrifices cross-domain robustness; open-set contin-
ued pretraining enhances generic transfer with small in-domain
trade-offs; open-vocabulary fine-tuning achieves the most fa-
vorable balance between precision on roadside categories and
generalization to broader distributions.
4.4.2. Attribute Recognition Comparision
We evaluate the performance of models in generating struc-
tured JSON outputs, and Table 8 presents the comparison re-
sults on two city-level datasets. The results demonstrate that
Qwen achieves 86.0% accuracy on the Shanghai dataset in the
zero-shot setting, reflecting its limited adaptation to roadside-
specific attributes. After fine-tuning and RAG, the attribute ac-
curacy on the Shanghai dataset improves to 95.5%, highlighting
the effectiveness of targeted domain adaptation for structured
multimodal outputs.
Beyond detection, we compare schema-constrained JSON at-
tribute outputs across several conversational vision–language
models (VLMs). As demonstrated in Table 8, the domain-
adapted, schema-guided Qwen model consistently achieves the
highest attribute accuracy on both city datasets, outperform-
13

Table 8
Accuracy evaluation of JSON-based QA outputs on Shanghai (SH) and
Wuhan (WH) datasets.
Method Strategy SH Acc WH Acc
Qwen-VL-Max Fine-tuned&RAG 95.5 94.1
Qwen-VL-Max Fine-tuned 92.7 88.4
Qwen-VL-Max Zero-shot 86.0 74.0
Claude3.7 Sonnet Zero-shot 75.2 71.5
GPT-4o Zero-shot 85.4 73.6
Llama 4 Maverick Zero-shot 70.8 56.1
ing general-purpose systems on fine-grained roadside attributes.
In particular, the fine-tuned Qwen-VL-Max with RAG attains
an attribute accuracy of 95.5% on the Shanghai dataset and
94.1% on the Wuhan dataset, substantiating the benefit of do-
main adaptation and schema-constrained prompting for reliable
structured outputs.
It is worth noting that the attribute recognition accuracy on
the Wuhan dataset is slightly lower than that on the Shanghai
dataset. This performance gap can be partially attributed to dif-
ferences in data acquisition conditions, including weather, illu-
mination, environmental complexity, and capture time, which
introduce additional visual ambiguity and variability in real-
world scenes.
Notably, the integration of a dual-modality Retrieval-
Augmented Generation (RAG) mechanism further enhances
reasoning consistency and domain grounding. During infer-
ence, thetextual RAGdynamically retrieves relevant segments
from professional knowledge bases, while thevisual RAGre-
trieves similar exemplars from the attribute-annotated image
repository. The retrieved textual definitions and visual refer-
ences are incorporated into the model’s multimodal context, en-
abling Qwen-VL to reason over both semantic and perceptual
evidence. This dual retrieval mechanism ensures higher seman-
tic fidelity, interpretability, and attribute completeness, particu-
larly for subtle distinctions in material, functionality, or regula-
tory meaning that purely visual cues cannot disambiguate.
Overall, the fine-tuned and RAG-enhanced Qwen model sur-
passes both zero-shot and non-knowledge-retrieval models on
both city datasets, confirming the effectiveness of combining
structured schema guidance with multimodal knowledge re-
trieval.
In summary, the combination of schema-constrained gen-
eration with multimodal retrieval significantly improves the
model’s ability to accurately recognize and interpret roadside
infrastructure attributes, even for complex or ambiguous cat-
egories. The fine-tuned Qwen model, augmented with RAG,
achieves robust performance across diverse urban scenarios,
confirming the advantage of this approach in generating high-
quality structured outputs.
5. Conclusions
This study unleashes the capabilities of Large Vi-
sion–Language Models (LVLMs) for the intelligent percep-
tion of urban roadside scenarios. By integrating an attribute-based schema with structured JSON generation and a dual-
modality Retrieval-Augmented Generation (RAG) mechanism,
we bridge the critical gap between unstructured visual descrip-
tions and practical decision support. The proposed framework
transforms generic perception capabilities into interpretable and
machine-readable solutions, directly empowering automated
infrastructure monitoring, maintenance, and operational plan-
ning.
Comprehensive experiments validated the effectiveness of
the proposed fine-tuning and enhancement strategies. Under
open-vocabulary fine-tuning, the detector achieved 58.9 mAP
on roadside categories and 47.6 mAP on COCO-style cate-
gories, demonstrating robust performance across both domain-
specific and general objects. On the multimodal reasoning
side, the domain-adapted Qwen-VL model—enhanced through
LoRA fine-tuning and augmented with a dual-modality knowl-
edge base constructed from professional textual references
(e.g., GB 5768.2–2022) and visual exemplars—achieved 95.5%
attribute accuracy. This verifies the reliability of schema-guided
and knowledge-grounded attribute interpretation.
Future work will extend this framework from static image
analysis to dynamic video understanding. We aim to incorpo-
rate temporal reasoning capabilities to support video-based ob-
ject detection and continuous state tracking. By analyzing dy-
namic changes over time, the system will be better equipped to
monitor the operational lifecycle of roadside assets and main-
tain robustness under evolving environmental conditions.
6. Declaration of Interests
The authors declare that they have no known competing fi-
nancial interests or personal relationships that could have ap-
peared to influence the work reported in this paper.
7. Declaration of Generative AI and AI-assisted technolo-
gies in the writing process
During the preparation of this work the author(s) used
used different large language models in order to polish the
manuscript and improve the readability. After using these
tool/service, the author(s) reviewed and edited the content as
needed and take(s) full responsibility for the content of the pub-
lication.
8. Acknowledgment
This work was jointly supported by the National Natural Sci-
ence Foundation of China (No. 42130105).
References
[1] N. Ma, J. Fan, W. Wang, J. Wu, Y . Jiang, L. Xie, R. Fan,
Computer vision for road imaging and pothole detec-
tion: a state-of-the-art review of systems and algorithms,
Transportation Safety and Environment 4 (2022) tdac026.
doi:10.1093/tse/tdac026.
14

[2] Z. Bai, G. Wu, X. Qi, Y . Liu, K. Oguchi, M. J. Barth,
Infrastructure-based object detection and tracking for co-
operative driving automation: A survey, in: 2022 IEEE In-
telligent Vehicles Symposium (IV), 2022, pp. 1366–1373.
doi:10.1109/IV51971.2022.9827461.
[3] A. Campbell, A. Both, Q. C. Sun, Detecting and
mapping traffic signs from google street view im-
ages using deep learning and gis, Computers, En-
vironment and Urban Systems 77 (2019) 101350.
URL:https://www.sciencedirect.com/science/
article/pii/S0198971519300870. doi:https://
doi.org/10.1016/j.compenvurbsys.2019.101350.
[4] C. Cui, Y . Ma, X. Cao, W. Ye, Y . Zhou, K. Liang, J. Chen,
J. Lu, Z. Yang, K.-D. Liao, et al., A survey on mul-
timodal large language models for autonomous driving
(2024) 958–979.
[5] L. Wen, X. Yang, D. Fu, X. Wang, P. Cai, X. Li, T. Ma,
Y . Li, L. Xu, D. Shang, et al., On the road with gpt-
4v (ision): Early explorations of visual-language model
on autonomous driving, arXiv preprint arXiv:2311.05332
(2023).
[6] J. Redmon, S. Divvala, R. Girshick, A. Farhadi, You only
look once: Unified, real-time object detection, in: 2016
IEEE Conference on Computer Vision and Pattern Recog-
nition (CVPR), 2016, pp. 779–788. doi:10.1109/CVPR.
2016.91.
[7] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed,
C.-Y . Fu, A. C. Berg, Ssd: Single shot multibox de-
tector, in: Proceedings of the 2016 European Con-
ference on Computer Vision (ECCV), 2016, pp. 21–37.
doi:10.1007/978-3-319-46448-0_2.
[8] C. Liu, M. Xie, C. Yuan, F. Liang, Z. Dong, B. Yang,
Training-free open-set 3d inventory of transportation in-
frastructure by combining point clouds and images, Au-
tomation in Construction 178 (2025) 106377. doi:https:
//doi.org/10.1016/j.autcon.2025.106377.
[9] Y . Zhou, X. Han, M. Peng, H. Li, B. Yang,
Z. Dong, B. Yang, Street-view imagery guided
street furniture inventory from mobile laser scan-
ning point clouds, ISPRS Journal of Photogramme-
try and Remote Sensing 189 (2022) 63–77. URL:
https://www.sciencedirect.com/science/
article/pii/S0924271622001265. doi:https:
//doi.org/10.1016/j.isprsjprs.2022.04.023.
[10] X. Han, C. Liu, Y . Zhou, K. Tan, Z. Dong, B. Yang, Whu-
urban3d: An urban scene lidar point cloud dataset for
semantic instance segmentation, ISPRS Journal of Pho-
togrammetry and Remote Sensing 209 (2024) 500–513.
URL:https://www.sciencedirect.com/science/
article/pii/S0924271624000522. doi:https:
//doi.org/10.1016/j.isprsjprs.2024.02.007.[11] L. H. Li, P. Zhang, H. Zhang, J. Yang, C. Li, Y . Zhong,
L. Wang, L. Yuan, L. Zhang, J.-N. Hwang, K.-W. Chang,
J. Gao, Grounded language-image pre-training, in: 2022
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2022, pp. 10955–10965. doi:10.
1109/CVPR52688.2022.01069.
[12] H. Liu, C. Li, Q. Wu, Y . J. Lee, Visual instruc-
tion tuning, in: A. Oh, T. Naumann, A. Glober-
son, K. Saenko, M. Hardt, S. Levine (Eds.), Ad-
vances in Neural Information Processing Sys-
tems, volume 36, Curran Associates, Inc., 2023,
pp. 34892–34916. URL:https://proceedings.
neurips.cc/paper_files/paper/2023/file/
6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.
pdf.
[13] H. Zhang, F. Li, S. Liu, L. Zhang, H. Su, J. Zhu,
L. Ni, H.-Y . Shum, Dino: Detr with improved denois-
ing anchor boxes for end-to-end object detection, in:
The Eleventh International Conference on Learning Rep-
resentations, 2023. URL:https://openreview.net/
forum?id=3mRwyG5one.
[14] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh,
S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark,
et al., Learning transferable visual models from natural
language supervision, in: International conference on ma-
chine learning, PmLR, 2021, pp. 8748–8763.
[15] T. Cheng, L. Song, Y . Ge, W. Liu, X. Wang, Y . Shan,
Yolo-world: Real-time open-vocabulary object detection,
in: 2024 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2024, pp. 16901–16911.
doi:10.1109/CVPR52733.2024.01599.
[16] K. Behrendt, L. Novak, R. Botros, A deep learning ap-
proach to traffic lights: Detection, tracking, and clas-
sification, in: 2017 IEEE International Conference on
Robotics and Automation (ICRA), IEEE Press, 2017, p.
1370–1377. doi:10.1109/ICRA.2017.7989163.
[17] O. Adedeji, Z. Wang, Intelligent waste classification
system using deep learning convolutional neural net-
work, Procedia Manufacturing 35 (2019) 607–612.
URL:https://www.sciencedirect.com/science/
article/pii/S2351978919307231. doi:https:
//doi.org/10.1016/j.promfg.2019.05.086, the
2nd International Conference on Sustainable Materials
Processing and Manufacturing, SMPM 2019, 8-10 March
2019, Sun City, South Africa.
[18] D. Tabernik, D. Sko ˇcaj, Deep learning for large-scale
traffic-sign detection and recognition, IEEE Transactions
on Intelligent Transportation Systems 21 (2020) 1427–
1440. doi:10.1109/TITS.2019.2913588.
[19] S. Houben, J. Stallkamp, J. Salmen, M. Schlipsing,
C. Igel, Detection of traffic signs in real-world im-
ages: The german traffic sign detection benchmark, in:
15

The 2013 International Joint Conference on Neural Net-
works (IJCNN), 2013, pp. 1–8. doi:10.1109/IJCNN.
2013.6706807.
[20] J. Choi, H. Lee, Real-time traffic light recognition
with lightweight state recognition and ratio-preserving
zero padding, Electronics 13 (2024). URL:https://
www.mdpi.com/2079-9292/13/3/615. doi:10.3390/
electronics13030615.
[21] N. Ma, J. Fan, W. Wang, J. Wu, Y . Jiang, L. Xie,
R. Fan, Computer vision for road imaging and pot-
hole detection: a state-of-the-art review of systems and
algorithms, Transportation Safety and Environment
4 (2022) tdac026. URL:https://doi.org/10.1093/
tse/tdac026. doi:10.1093/tse/tdac026.
[22] Z. Aygün, M. Kocaman, S. Aydemir, B. Konako ˘glu,
Building damage detection using deep learning architec-
ture with satellite images: The case of the 6 february 2023
kahramanmara¸ s earthquake, International Journal of Pi-
oneering Technology and Engineering 3 (2024) 53–61.
doi:10.56158/jpte.2024.94.3.02.
[23] F. B. et al., An introduction to vision-language mod-
eling, 2024. URL:https://arxiv.org/abs/2405.
17247.arXiv:2405.17247.
[24] O. et al., Gpt-4 technical report, 2024. URL:https://
arxiv.org/abs/2303.08774.arXiv:2303.08774.
[25] H. Liu, C. Li, Q. Wu, Y . J. Lee, Visual instruc-
tion tuning, in: A. Oh, T. Naumann, A. Glober-
son, K. Saenko, M. Hardt, S. Levine (Eds.), Ad-
vances in Neural Information Processing Sys-
tems, volume 36, Curran Associates, Inc., 2023,
pp. 34892–34916. URL:https://proceedings.
neurips.cc/paper_files/paper/2023/file/
6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.
pdf.
[26] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin,
C. Zhou, J. Zhou, Qwen-vl: A versatile vision-language
model for understanding, localization, text reading, and
beyond, 2023. URL:https://arxiv.org/abs/2308.
12966.arXiv:2308.12966.
[27] S. Zhang, D. Fu, W. Liang, Z. Zhang, B. Yu,
P. Cai, B. Yao, Trafficgpt: Viewing, process-
ing and interacting with traffic foundation mod-
els, Transport Policy 150 (2024) 95–105. URL:
https://www.sciencedirect.com/science/
article/pii/S0967070X24000726. doi:https:
//doi.org/10.1016/j.tranpol.2024.03.006.
[28] H. Liu, W. Xue, Y . Chen, D. Chen, X. Zhao, K. Wang,
L. Hou, R. Li, W. Peng, A survey on hallucination in large
vision-language models, 2024. URL:https://arxiv.
org/abs/2402.00253.arXiv:2402.00253.[29] M. Maaz, H. Rasheed, S. Khan, F. Khan, Video-chatgpt:
Towards detailed video understanding via large vision and
language models, in: Proceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics
(V olume 1: Long Papers), 2024, pp. 12585–12602.
[30] J. Chen, B. Lin, R. Xu, Z. Chai, X. Liang, K.-Y . Wong,
Mapgpt: Map-guided prompting with adaptive path plan-
ning for vision-and-language navigation, 2024.
[31] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin,
C. Zhou, J. Zhou, Qwen-vl: A frontier large vision-
language model with versatile abilities, arXiv preprint
arXiv:2308.12966 1 (2023) 3.
[32] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang,
L. Wang, W. Chen, et al., Lora: Low-rank adaptation of
large language models, 2022.
[33] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel,
et al., Retrieval-augmented generation for knowledge-
intensive nlp tasks, Advances in Neural Information Pro-
cessing Systems (NeurIPS) 33 (2020) 9459–9474.
[34] S. Pan, L. Luo, Y . Wang, C. Chen, J. Wang, X. Wu, Uni-
fying large language models and knowledge graphs: A
roadmap, IEEE Transactions on Knowledge and Data
Engineering 36 (2024) 3580–3599. doi:10.1109/TKDE.
2024.3352100.
[35] Y . Zhou, X. Li, Q. Wang, J. Shen, Visual in-context
learning for large vision-language models, in: Find-
ings of the Association for Computational Linguis-
tics: ACL 2024, Association for Computational
Linguistics, 2024, pp. 15890–15902. URL:https:
//aclanthology.org/2024.findings-acl.940.
doi:10.18653/v1/2024.findings-acl.940.
[36] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. En-
zweiler, R. Benenson, U. Franke, S. Roth, B. Schiele, The
cityscapes dataset for semantic urban scene understand-
ing, in: Proceedings of the IEEE conference on computer
vision and pattern recognition, 2016, pp. 3213–3223.
[37] F. Yu, H. Chen, X. Wang, W. Xian, Y . Chen, F. Liu,
V . Madhavan, T. Darrell, Bdd100k: A diverse driving
dataset for heterogeneous multitask learning, in: Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, 2020, pp. 2636–2645.
[38] T.-Y . Lin, M. Maire, S. Belongie, J. Hays, P. Perona,
D. Ramanan, P. Dollár, C. L. Zitnick, Microsoft coco:
Common objects in context, in: Computer Vision – ECCV
2014, Springer International Publishing, Cham, 2014, pp.
740–755.
[39] H. Wang, P. Ren, Z. Jie, X. Dong, C. Feng, Y . Qian,
L. Ma, D. Jiang, Y . Wang, X. Lan, et al., Ov-dino: Unified
open-vocabulary detection with language-aware selective
fusion, arXiv preprint arXiv:2407.07844 (2024).
16

[40] S. Shao, Z. Li, T. Zhang, C. Peng, G. Yu, X. Zhang, J. Li,
J. Sun, Objects365: A large-scale, high-quality dataset for
object detection, in: Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision (ICCV), 2019,
pp. 8430–8439.
[41] P. Sharma, N. Ding, S. Goodman, R. Soricut, Conceptual
captions: A cleaned, hypernymed, image alt-text dataset
for automatic image captioning, in: Proceedings of the
56th Annual Meeting of the Association for Computa-
tional Linguistics (ACL), 2018, pp. 2556–2565.
17