# From Pixels to Posts: Retrieval-Augmented Fashion Captioning and Hashtag Generation

**Authors**: Moazzam Umer Gondal, Hamad Ul Qudous, Daniya Siddiqui, Asma Ahmad Farhan

**Published**: 2025-11-24 14:13:57

**PDF URL**: [https://arxiv.org/pdf/2511.19149v1](https://arxiv.org/pdf/2511.19149v1)

## Abstract
This paper introduces the retrieval-augmented framework for automatic fashion caption and hashtag generation, combining multi-garment detection, attribute reasoning, and Large Language Model (LLM) prompting. The system aims to produce visually grounded, descriptive, and stylistically interesting text for fashion imagery, overcoming the limitations of end-to-end captioners that have problems with attribute fidelity and domain generalization. The pipeline combines a YOLO-based detector for multi-garment localization, k-means clustering for dominant color extraction, and a CLIP-FAISS retrieval module for fabric and gender attribute inference based on a structured product index. These attributes, together with retrieved style examples, create a factual evidence pack that is used to guide an LLM to generate human-like captions and contextually rich hashtags. A fine-tuned BLIP model is used as a supervised baseline model for comparison. Experimental results show that the YOLO detector is able to obtain a mean Average Precision (mAP@0.5) of 0.71 for nine categories of garments. The RAG-LLM pipeline generates expressive attribute-aligned captions and achieves mean attribute coverage of 0.80 with full coverage at the 50% threshold in hashtag generation, whereas BLIP gives higher lexical overlap and lower generalization. The retrieval-augmented approach exhibits better factual grounding, less hallucination, and great potential for scalable deployment in various clothing domains. These results demonstrate the use of retrieval-augmented generation as an effective and interpretable paradigm for automated and visually grounded fashion content generation.

## Full Text


<!-- PDF content starts -->

From Pixels to Posts: Retrieval-Augmented Fashion Captioning and Hashtag Generation
Moazzam Umer Gondala,∗, Hamad Ul Qudousa,∗, Daniya Siddiquia, Asma Ahmad Farhana
aSchool of Computing, National University of Computer&Emerging Sciences (FAST), Lahore, 54000, Pakistan
Abstract
This paper introduces the retrieval-augmented framework for automatic fashion caption and hashtag generation, combining multi-
garment detection, attribute reasoning, and Large Language Model (LLM) prompting. The system aims to produce visually
grounded, descriptive, and stylistically interesting text for fashion imagery, overcoming the limitations of end-to-end captioners
that have problems with attribute fidelity and domain generalization. The pipeline combines a YOLO-based detector for multi-
garment localization, k-means clustering for dominant color extraction, and a CLIP-FAISS retrieval module for fabric and gender
attribute inference based on a structured product index. These attributes, together with retrieved style examples, create a factual
evidence pack that is used to guide an LLM to generate human-like captions and contextually rich hashtags. A fine-tuned BLIP
model is used as a supervised baseline model for comparison. Experimental results show that the YOLO detector is able to ob-
tain a mean Average Precision (mAP@0.5) of 0.71 for nine categories of garments. The RAG-LLM pipeline generates expressive
attribute-aligned captions and achieves mean attribute coverage of 0.80 with full coverage at the 50% threshold in hashtag genera-
tion, whereas BLIP gives higher lexical overlap and lower generalization. The retrieval-augmented approach exhibits better factual
grounding, less hallucination, and great potential for scalable deployment in various clothing domains. These results demonstrate
the use of retrieval-augmented generation as an effective and interpretable paradigm for automated and visually grounded fashion
content generation.
Keywords:RAG, LLM, YOLO, CLIP, BLIP, FAISS, Fashion Captioning
1. Introduction
Fashion images on social media frequently depict multiple
garments styled together and captured in unconstrained con-
ditions, where composition, lighting, and pose are optimized
for human appeal rather than machine perception. The cen-
tral problem is to automatically produce a caption and a set
of hashtags that are faithful to the visual evidence: the text
must name the detected garment categories, correctly reflect
salient attributes such as dominant colors and likely fabric, and
adopt language that suits social platforms and brand tone. De-
spite steady progress in generic image captioning, recent sur-
veys document persistent difficulty with fine-grained attribute
grounding and multi-object compositionality, which leads to
omissions or hallucinated object–attribute pairings [1, 2]. For
example, when two similarly textured garments co-occur, cap-
tioners may collapse them into a single mention or attach the
wrong color term to the wrong item; when accessories are par-
tially occluded, they may be ignored or spuriously introduced
[3]. In creator and brand workflows, such errors translate into
captions that read well but fail to reflect what is actually in the
photo, and into hashtags that skew generic rather than represen-
tative of color, material, or audience [4, 5].
∗These authors contributed equally to this work.
Email addresses:moazzamumar22@gmail.com(Moazzam Umer
Gondal),hamad.ulqudous@nu.edu.pk(Hamad Ul Qudous),
daniyasiddiqui120@gmail.com(Daniya Siddiqui),
asma.ahmad@nu.edu.pk(Asma Ahmad Farhan)The significance of this issue is that captions and hashtags
serve as light-weight, consumer-facing metadata and have a di-
rect impact on search, recommendations, and audience target-
ing in fashion ecosystems. Offering industry-facing analyses,
it is claimed that social streams are one of the most important
leading indicators of trend dynamics and merchandising sig-
nals, and this has a definite implication on planning and content
strategy [6]. Tag suggestion studies show that context-aware
and structured tags enhance discoverability and engagement
through matching user intent with platform discovery and rank-
ing of tags [5]. A similar role is played by retrieval throughout
the fashion pipelines: product discovery relies on visual simi-
larity search, near-duplicate curation, and attribute propagation,
and retrieval offers a natural interface between perception and
language supervision when explicit labels are sparse or noisy
[7]. In practice, the brands have to work both in the studio
catalog imagery and in-the-wild social posts, a deployable so-
lution needs to have strong perception of a variety of garments
and local colors, and controlled generation that can give short
and style-consistent copy and balanced mix of broad and niche
tags, at latencies that fit within real-time or near-real-time con-
tent processes.
Literature shows a trajectory from CNN–RNN en-
coder–decoders toward transformer-based vision–language
models. Surveys catalogue advances in attention mechanisms,
large-scale pretraining, and decoding strategies, while noting
weaknesses in attribute specificity and compositional gener-
alization that surface in complex scenes [1, 2]. Methods thatarXiv:2511.19149v1  [cs.CV]  24 Nov 2025

enrich the visual representation with relational structure (e.g.,
scene graphs) mitigate certain error modes by making ob-
ject–relation patterns explicit; nonetheless, they often struggle
to bind the correct attribute tokens to the correct instances when
similar items co-occur or partially occlude one another [3]. In
fashion-specific settings, works that explicitly align detected
visual attributes with language—conditioning generation on
item categories and attributes—report improved faithfulness
and reduced drift from the image content, suggesting attribute
grounding is central to domain realism [8]. Complementary
research treats captioning as a retrieval-informed task, retriev-
ing semantically close exemplars or summaries to stabilize
wording, reduce hallucination, and better reflect the target
register and platform constraints [9]. On the social side, studies
of caption and hashtag recommendation reinforce the value of
controlled phrasing and structured tags for visibility, intent-
signaling, and downstream analytics [4, 5]. Together, these
threads motivate designs that unify multi-object perception,
explicit attribute extraction, and retrieval-informed generation,
rather than relying on a single monolithic captioner.
Addressing this problem in practice introduces several chal-
lenges that shape the design space. First, multi-garment per-
ception is intrinsically difficult: garments overlap, self-occlude,
and deform, and recall must be preserved without admitting
spurious detections that pollute downstream text. Analyses of
modern one-stage detectors highlight sensitivity to confidence
thresholds, non-maximum suppression, and class imbalance,
underscoring a precision–recall trade-offthat must be carefully
managed in fashion scenes [10, 11]. Second, attribute fidelity
depends on localized evidence: dominant colors should be com-
puted on garment crops rather than the whole image to avoid
background bias, and fine-grained cues such as likely fabric or
audience frequently require auxiliary signals beyond the detec-
tor’s category label. Fine-grained garment segmentation and
fashion classification studies emphasize the value of instance-
or part-level delineation to recover attributes that are visually
subtle but semantically important [12, 13]. Third, domain shift
between studio catalog photos and social posts degrades out-of-
domain performance; cross-domain evaluations regularly report
substantial drops without explicit adaptation or retrieval-based
conditioning [14]. Finally, deployment requires robustness test-
ing and diagnostics to surface failure modes such as attribute
hallucination, missed detections, or brittle decoding behaviors,
so that thresholds and safeguards can be tuned for real settings
[15].
We address these challenges with a retrieval-augmented for-
mulation that modularizes perception and generation, passing
structured evidence between them. Our approach integrates
multi-object detection to identify garment instances, localized
color estimation to anchor color words, and retrieval against a
curated catalog to supply stable attribute cues and stylistic ex-
amples. This modular framework allows the system to generate
fluent captions and diverse hashtags by conditioning on a com-
pact evidence pack that enumerates detected categories, local-
ized colors, and retrieved attributes. Importantly, the retrieval
step serves as a stability prior, reducing attribute hallucination
and ensuring diversity in linguistic output without requiring thelanguage generator to be retrained.
To the best of our knowledge, this work is the first to focus
on South Asian fashion using a pipeline that combines multi-
garment detection, color extraction, and retrieval-augmented
captioning. We present a unique dataset and framework tai-
lored to address the distinctive attributes of South Asian ap-
parel, an area that has been underrepresented in contemporary
vision–language research and training corpora.
We make the following contributions within the fashion cap-
tioning domain:
•A unified, deployment-oriented pipeline that integrates
multi-garment perception, localized color extraction, and
retrieval-informed language generation for fashion social
media, designed to reduce attribute hallucination and im-
prove the coverage of color, fabric, and target audience.
•A new annotated dataset of over 2,500 South Asian ap-
parel images, curated for multi-garment detection and cap-
tioning research, addressing a region and style underrepre-
sented in current vision–language datasets.
•A retrieval module that consolidates weak attributes via
similarity-weighted voting and supplies near-neighbor
style hints, improving robustness across domain shifts be-
tween catalog imagery and social media content.
•An empirical comparison with a supervised captioning
baseline trained on our dataset, highlighting the strengths
and limitations of retrieval-augmented generation in the
context of a fashion domain that remains underexplored
in the literature, accompanied by robustness diagnostics
inspired by modern testing frameworks.
2. Related Work
Image captioning has evolved into a central task in vi-
sion–language research, aiming to translate visual informa-
tion into coherent natural language. Over the past decade,
progress in deep learning has transformed caption generation
from template-based and retrieval-driven methods into data-
driven architectures that learn cross-modal alignments between
vision and text. Early frameworks relied on convolutional and
recurrent neural networks to extract image features and sequen-
tially decode captions, while recent transformer-based models
leverage self-attention to capture richer global context and long-
range dependencies. Surveys of the field describe this evolu-
tion and highlight how attention mechanisms, pre-trained vi-
sion–language encoders, and multimodal embeddings have sig-
nificantly improved fluency and contextual relevance in gener-
ated captions [1, 2]. Despite these advances, challenges per-
sist in fine-grained attribute grounding, object–relation reason-
ing, and domain adaptation, particularly in specialized domains
such as fashion or scientific imagery. The broader literature
now integrates hybrid pipelines, retrieval augmentation, and
domain-specific modeling to address these gaps, reflecting a
growing effort to bridge perception accuracy with linguistic ex-
pressiveness in modern captioning systems.
2

The evolution of image captioning has progressed from early
encoder–decoder architectures toward transformer-based mul-
timodal frameworks. Classical models coupled convolutional
neural networks with recurrent decoders to map image fea-
tures into textual sequences, achieving foundational success on
datasets such as MS COCO and Visual Genome [1, 16]. At-
tention mechanisms were later integrated to dynamically fo-
cus on relevant visual regions during word generation, substan-
tially improving alignment between localized objects and corre-
sponding linguistic tokens [2]. These developments established
the core paradigm for automatic caption generation and led to
measurable gains across standard evaluation metrics including
BLEU, METEOR, and CIDEr [1]. Recent transformer-based
architectures further advanced this field by modeling global de-
pendencies through self-attention, enabling richer context un-
derstanding and more coherent sentence generation across di-
verse domains [16, 2].
Beyond architectural evolution, several frameworks sought
to enhance semantic granularity by decomposing visual scenes
into localized entities. Dense-CaptionNet introduced a region-
based captioning pipeline that first generated object-level and
attribute-level descriptions before merging them into compre-
hensive sentences, resulting in more accurate portrayals of com-
plex or cluttered images [17]. The design allowed simulta-
neous modeling of multiple regions and relationships, outper-
forming earlier holistic captioners on datasets such as Visual
Genome and MS COCO. Similarly, multi-network ensembles
that combine multiple convolutional encoders with natural lan-
guage modules have been proposed to reduce the semantic gap
between visual perception and sentence generation. By fusing
diverse feature maps prior to decoding, these systems generate
richer and more detailed captions, demonstrating improved de-
scriptive precision over single-model approaches [18].
Despite these advances, surveys consistently report persistent
issues such as object hallucination, missing attributes, and lim-
ited transferability to unseen domains [1, 16, 2]. Furthermore,
conventional n-gram metrics often fail to capture semantic fi-
delity and human judgment of caption quality, motivating the
exploration of alternative evaluation measures and more struc-
tured visual–language alignment strategies [2]. Collectively,
these studies highlight both the achievements and limitations
of generic captioning systems and establish the technical foun-
dation for recent domain-specific and retrieval-augmented ap-
proaches that aim to generate attribute-grounded and contextu-
ally faithful captions.
An expanding literature is modifying image captioning mod-
els to specific fields where visual semantics and vocabulary are
not the same as in everyday images. These domain-based sys-
tems alter encoders, decoders or datasets to obtain fine-grained
details as well as professional language. In fashion, the abil-
ity to incorporate structured attribute data in captioning has
shown a great deal of success in enhancing descriptive and
stylistic relevance. An attribute alignment module was pre-
sented in an attribute study, and this links grid-level image fea-
tures to annotated clothing attributes and a fashion language
model trained on a balanced corpus to maintain less vocabu-
lary bias. This combination generated a variety of attribute-based captions and did better with fashion datasets FACAD and
Fashion-Gen, proving that the generation of text based on visual
attributes improves fidelity and contextual richness [8].
The similar approach can be found in other specialized ar-
eas that demand technical accuracy. A hybrid model of con-
volutional feature extraction with word embeddings trained on
geoscience literature is used in geological image captioning,
where the decoder is encouraged to use mineralogical terminol-
ogy through the choice of words. Domain-specific embeddings
enhanced semantic accuracy and linguistic suitability compared
to generic baselines through integration with domain-specific
embeddings, as well as both semantic and linguistic accuracy
and appropriateness scores [19]. An adaptive attention model
in civil engineering that was modified to bridge inspection im-
ages focused on areas with damage, e.g., cracks or corrosion,
and generated captions that were similar to engineer evaluations
both in terminology and spatial resolution to the damage area
[20]. These methods indicate that the correspondence of at-
tention mechanisms and vocabularies to domain cues generates
descriptions that satisfy experts.
Other studies generalize captioning to situations that need
general spatial reasoning. A remote sensing pipeline that used
U-Net to segment and then caption a scene enhanced the inter-
pretation of the scene, concentrating attention on meaningful
land-use categories, resulting in more informative and under-
standable descriptions of aerial imagery [21]. A comparison of
Vision Transformer and VGG encoders with satellite caption-
ing revealed that ViT global self-attention enhances contextual
consistency and boosts BLEU and CIDEr scores compared to
CNN-based features [22]. Cross-domain tests also demonstrate
the sensitivity of architecture: a comparative study of fashion,
art, and medical data indicated that LSTM-based decoders are
more likely to be consistent in generalizing to heterogeneous
types of images than purely Transformer-based decoders do
when adapting to heterogeneous data intrinsically [14]. Taken
together, these works confirm that the domain adaptation, at-
tribute grounding, and adaptive attention mechanisms signif-
icantly enhance caption quality, which provides a theoretical
basis of creating retrieval-enhanced fashion captioning systems,
which have to acquire localized attributes and stylistic subtlety.
Accurate captioning in the fashion domain depends on reli-
able visual perception of garments, making detection and seg-
mentation essential. Modern one-stage detectors such as the
YOLO family are widely used for their ability to localize mul-
tiple objects in real time while maintaining strong accuracy.
Studies describe how YOLO reformulates detection as a single
regression problem that directly predicts bounding boxes and
class probabilities from full images, enabling fast end-to-end
inference [10]. Comparative analyses across YOLO versions
highlight improvements such as stronger backbone networks,
anchor-free designs, and refined non-maximum suppression,
enhancing precision even under limited data or hardware re-
sources [11]. These advances make YOLO particularly suitable
for fashion scenarios involving multiple garments, accessories,
or patterns within a single frame.
Building on these foundations, domain-specific detection
models have been introduced for fashion imagery. A study
3

using YOLOv5 demonstrated efficient real-time detection of
clothing styles—such as plaid, plain, and striped patterns—on
modest hardware while maintaining high mean average preci-
sion. The results confirmed YOLO’s suitability for fine-grained
fashion recognition, showing that single-stage detectors can
balance accuracy and speed even on large-scale image collec-
tions [13]. Complementary work on garment segmentation ex-
tends this detection paradigm by delineating item boundaries at
pixel level. A modified Mask R-CNN with multi-scale feature
fusion and residual modules improved segmentation of overlap-
ping apparel and complex poses, achieving higher accuracy and
cleaner boundaries than the baseline model [12]. Similar se-
mantic segmentation enhancements, including edge-aware met-
rics and architecture adaptation methods, further refine region
boundaries and object shapes, enabling more precise extraction
of garment masks [23, 24].
Integration of detection and captioning has also been ex-
plored outside fashion. A vision-based system for construc-
tion imagery combined YOLO-style detection with a caption
generator to produce structured scene descriptions, demonstrat-
ing that localized object features enhance textual understanding
[25]. These findings collectively underscore that robust detec-
tion and segmentation modules form the perceptual backbone
of any captioning pipeline. For fashion captioning, they provide
the essential groundwork for isolating multiple garments, cap-
turing localized color information, and ensuring that language
generation aligns with the visual evidence present in each re-
gion.
The current studies are showing a tendency to incorporate re-
trieval and hybrid learning methods in the process of image cap-
tioning in order to enhance semantic consistency and ground-
ing. These methods are integrated visual recognition, text sum-
mary, and optimization to bridge the gap between perception
and language generation. A representative research incorpo-
rated both summarization and captioning, which involved the
integration of BiLSTM-based text encoding and Deep Belief
Network to summarize and produce image descriptions concur-
rently. This multimodal scheme boosts visual context in re-
trieval outcomes, and the accuracy, recall, and F-scores of sum-
marization and BLEU scores are not far below the human ones,
proving that cross-modal fusion increases semantic richness in
retrieval-oriented tasks [9].
Models based on optimization expand on this combination by
using metaheuristic algorithms to optimize network parameters
or the output of the generated networks. A hybrid captioning
framework used genetic or particle swarm optimization to opti-
mize deep encoder-decoder models and optimize captions most
likely to be descriptive, providing quantifiable and statistically
significant improvements in BLEU and CIDEr in comparison
to baseline networks. The model was able to generate captions
that were syntactically consistent and semantically elaborate by
escaping local optima through metaheuristics application and
considering the model to be semantically detailed and syntacti-
cally well-formed in captioning [26]. Likewise, the ensemble-
based and item-level hybrid methods focus on the integration
of specialized feature extractors and attribute reasoning to em-
power caption semantics. A style classification network basedon item region, such as, made use of domain-specific pooling
and dual backbones to locate garment area and combine their
characteristics using gating processes, enhancing classification
precision by up to 17% compared to baselines [27]. Such ad-
vances of item-level feature representation guide captioning ar-
chitecture, which needs to formulate several visual objects with
exact associations.
The retrieval as the concept has become one of the cen-
tral elements of fashion and visual understanding mechanisms.
Surveys on fashion image retrieval classify the current sys-
tems as cross-domain, attribute-based, or outline-level retrieval
pipelines and refer to the fact that visual similarity embeddings
allow their application in product matching and complimentary
item recommendation applications [7]. Supplementary struc-
tures combine retrieval and generative language frameworks in
the individual styling. A generative AI-based recommenda-
tion system was used to generate textual advice about the out-
fits by hybridizing YOLOv8 detection with the GPT-4 model
and achieved good scores in evaluation and user satisfaction
in fashion recommendation tasks with localized clothing crops
[28]. Combined with these hybrid and retrieval-augmented
studies, one can conclude that structured search, optimization,
and attribute-level reasoning can generate more informative
and context-consistent captions, and is a conceptual basis of
retrieval-augmented generation pipelines.
Parallel to advances in vision–language modeling, research
has explored caption and hashtag generation for social media,
where engagement and contextual relevance are central objec-
tives. These studies combine computer vision, natural language
processing, and trend analytics to generate audience-aware tex-
tual content aligned with visual cues. A deep learning cap-
tion recommendation engine trained on Instagram data demon-
strated that integrating visual analysis with neural language
generation produces more contextually relevant captions than
manual authoring. The system used a convolutional network for
image understanding and a language model tuned to social me-
dia phrasing, capturing stylistic tone typical of platform com-
munication [4]. Complementary work on hashtag prediction
proposed a two-stage framework: a ResNet-based classifier de-
tected semantic concepts in images, and a transformer genera-
tor produced trending hashtags. This multimodal pipeline in-
creased relevant tag coverage by roughly 11% compared with
baseline approaches, confirming that coupling visual and lin-
guistic representations enhances content visibility [5].
Social analytics frameworks extend these ideas to large-scale
trend discovery. A fashion intelligence system applied object
detection to social media posts, focusing on handbags as a case
study, and extracted features such as type and dominant color
to identify emerging style patterns. Achieving 97% classifica-
tion accuracy and 0.77 mean average precision, it demonstrated
how automated analysis of social imagery can inform design
and marketing decisions [6]. Beyond caption and tag gener-
ation, multimodal discourse modeling examines how images
and text interact semantically. A study of cross-modality dis-
course classified five types of image–text relations—from direct
description to conceptual extension—using a multi-head atten-
tion model trained on annotated tweet pairs, achieving state-of-
4

the-art accuracy [29]. Collectively, these approaches illustrate
that blending image understanding with language modeling en-
hances the creativity, interpretability, and analytic value of so-
cial media content, reinforcing the need for captioning systems
that balance visual precision with communicative engagement.
Despite extensive progress across captioning, detection, and
retrieval research, several limitations remain that constrain
real-world deployment in fashion applications. Most existing
captioning models, including transformer-based and domain-
adapted variants, focus on single-object scenes or rely on
pre-defined attributes, limiting their ability to describe multi-
garment compositions typical of social media fashion imagery.
Detection-oriented studies excel in localizing objects but rarely
connect those results to coherent language generation, while
retrieval and optimization frameworks often address seman-
tic alignment in isolation rather than integrating visual per-
ception with generative modeling. Current domain-specific
works improve attribute grounding but are confined to narrow
datasets and lack mechanisms to generalize stylistically across
domains. Furthermore, social-media captioning systems prior-
itize engagement or tag relevance without ensuring factual cor-
respondence to image content. Collectively, these gaps high-
light the need for an integrated pipeline that unifies multi-object
detection, localized attribute extraction, and retrieval-informed
language generation to produce accurate, context-aware, and
stylistically adaptive captions for complex fashion scenes.
The remainder of this paper is organized as follows. Sec-
tion 3 outlines the proposed methodology, including data prepa-
ration, model design, and retrieval-augmented caption genera-
tion. Section 4 describes the implementation setup, experimen-
tal configuration, and training details for all components. Sec-
tion 5 presents the quantitative and qualitative evaluation of the
proposed system, covering detection, captioning, and hashtag
generation. Section 6 discusses the key findings, limitations,
and directions for future work, while Section 7 concludes the
paper with final remarks and implications.
3. Methodology
The suggested framework unites the concepts of multi-
garment recognition, retrieval-enhanced rationale, and gener-
ative caption creation into one fashion image comprehension
structure. The system has been structured into three signifi-
cant stages as demonstrated in Fig. 1. In theObject Detec-
tionblock, a YOLO-based model detects all garments in the
image (e.g., shirt, dupatta, frock, cordset) and labels each of
them with a class. TheInformation Retrievalblock calculates
embeddings of the identified image and retrieves semantically
similar examples, integrating the visual evidence by a focused
embedding encoding the fabric and gender features. Lastly,
theGenerationblock makes use of an LLM to turn the struc-
tured evidence into fluent but attribute-based captions and gen-
erates a uniform set of hashtags. Together, these modules form
a retrieval-augmented pipeline capable of generating visually
faithful and stylistically adaptive social media descriptions.
Figure 1: High-level architecture of the proposed retrieval-augmented fashion
captioning pipeline showing object detection, information retrieval, and gener-
ative captioning stages.
3.1. Data Preparation and Preprocessing
The data utilized in the present work was collected in the
form of posts in social media and web pages of fashion brands
that are publicly available in order to include consumer-style
and catalog-quality images. The corpus collected is varied in
terms of both category of clothes, poses and lighting conditions,
which mirror the diversity of fashion photography in the real
world. There is also a subset of pictures that contain metadata,
as the titles of a product or a short description, and there are also
pictures that contain only raw images. All pictures were down-
sized to a standard resolution, placed in the RGB color space
and filtered to exclude duplicates or low-quality samples. In
the case of metadata, text fields were cleansed and tokenized
to identify useful properties such as color, gender, and fab-
rics. The resulting dataset thus integrates multi-source visual
data and heterogeneous textual information, forming a balanced
foundation for training and evaluating the proposed detection,
retrieval, and caption generation modules under varied stylistic
and environmental conditions.
3.2. Object Detection and Visual Attribute Extraction
The perception stage of the system is responsible for identi-
fying all visible garments in a fashion image. A YOLO-based
object detector was fine-tuned on the curated fashion dataset
comprising categories such as shirt, dupatta, trouser, frock,
and other garments. The model was trained to detect multi-
ple garments per image, enabling multi-label and multi-instance
recognition under varied poses and lighting conditions. Dur-
ing inference, all bounding boxes with confidence scores above
a fixed threshold (θ con f) and non-maximum suppression IoU
threshold (θ iou) are retained to ensure that overlapping garments
5

Table 1: Condensed summary of representative literature discussed in Sections 2.1–2.5.
# Domain/Task Core Dataset/Con-
textMethod and Key Highlights
[1] Generic captioning
(survey)MS COCO, Visual
GenomeTaxonomy of CNN–RNN, attention, transformers; datasets & metrics; challenges incl. hallucination
and weak attribute grounding.
[16] Captioning (survey) Multiple benchmarks Structured review; taxonomy and comparative ranking; highlights bias, misalignment, interpretability.
[17] Region-based caption-
ingVisual Genome, MS
COCODense-CaptionNet: region/object and attribute description fused to full sentence; improved detail on
complex scenes.
[18] Captioning (ensemble) Generic image
datasetsMulti-network CNN ensemble+NLP decoder; fuses diverse visual features to reduce semantic gap and
boost precision.
[2] Captioning (survey) Multiple benchmarks Trends from CNN–RNN to Transformers; attention, scene structure; evaluation–human mismatch and
real-time constraints.
[8] Fashion captioning Fashion-Gen, FACAD Attribute Alignment Module+Fashion Language Model (balanced corpus); improves attribute-
grounded, diverse captions.
[19] Geological captioning Domain-specific geol-
ogyCNN+domain word embeddings; injects scientific vocabulary; more accurate, context-appropriate
terminology.
[20] Civil engineering Bridge inspection im-
agesAdaptive attention emphasizes damage regions; captions consistent with engineer reports (loca-
tion/type).
[21] Remote sensing cap-
tioningAerial imagery U-Net segmentation before captioning; focuses on meaningful regions; clearer land-use descriptions.
[22] Remote sensing Satellite datasets ViT vs VGG encoders; ViT’s global context improves BLEU/CIDEr and descriptive coherence.
[14] Cross-domain caption-
ingFashion, art, medical,
newsComparative study; LSTM decoders generalize better than pure Transformer decoders across domains.
[10] Object detection (re-
view)Real-time detection YOLO single-pass regression; high-speed multi-object localization and practical optimization insights.
[11] Detection/segmentation
(review)COCO evaluations Evolution of YOLO versions; backbone upgrades, anchor-free ideas, NMS tuning; precision under
constraints.
[13] Fashion style recogni-
tionCustom (five patterns) YOLOv5s detects plaid/plain/striped styles; high mAP and FPS on modest hardware (real-time feasi-
bility).
[12] Garment segmentation Fashion datasets Modified Mask R-CNN with multi-scale fusion/residual modules; cleaner boundaries for overlapping
apparel.
[23] Segmentation (edge-
aware)Generic segmentation Region-edge metric and loss to improve boundary quality;+1% overall,+4% on edge-region metrics.
[24] Segmentation
(methodology)Automotive images Procedure to select/adapt encoder–decoder by task constraints;∼80% accuracy with short training.
[25] Detection+captioning Construction site im-
agesDetector features fed to captioner; structured captions (scene graphs) for queryable site understanding.
[9] Retrieval+captioning Gigaword, DUC BiLSTM text encoding+DBN summarization+image captions; better P/R/F for summaries; captions
near human BLEU.
[26] Hybrid optimization Standard caption sets Metaheuristics (GA/PSO) tune encoder–decoder and outputs; BLEU/CIDEr gains via search beyond
local optima.
[27] Fashion style classifi-
cationFashion style sets Item-region pooling, dual backbones, gated fusion; up to 16–17% accuracy gains (avg.∼9%).
[7] Fashion image re-
trieval (survey)Cross-/attribute/outfit
retrievalTaxonomy of FIR; cross-domain matching, attribute-based search, outfit recommendation; multimodal
embedding needs.
[28] Fashion recommenda-
tionUser photos;
YOLOv8+LLMYOLO crops+GPT-based advice; competitive user ratings vs other assistants; localized, personalized
styling.
[4] Social captioning Instagram data CNN+NLP caption engine; platform-aligned tone; more contextually suitable than manual authoring.
[5] Hashtag generation Social images ResNet classifier+Transformer generator;∼11% increase in relevant/trending tags over baselines.
[6] Trend mining Instagram (handbags) Detects handbags; extracts type/colors; 97% classification, mAP 0.77; dashboards for trend discovery.
[29] Image–text discourse 16k tweets (labeled) Multi-head attention classifier of five image–text relations; state-of-the-art discourse relation accuracy.
are accurately localized. Each detected region is then cropped
to serve as an independent visual instance for further analysis.
For every cropped detection, the dominant garment colors are
estimated usingk-means clustering in the RGB space. Given
pixel samples{x 1,x2,...,x n}, the algorithm partitions them into
kclusters{C 1,C2,...,C k}by minimizing intra-cluster variance:
arg min
CkX
i=1X
xj∈Ci∥xj−µ i∥2,(1)
whereµ irepresents the centroid of clusterC i. The two largest
clusters are chosen as primary and secondary colors after dis-carding near-white or near-black clusters. Each centroid is
mapped to its nearest perceptual color name in the CIELAB
color space using Euclidean distance, providing an interpretable
and consistent color representation for each garment. The re-
sulting detections and associated color features form the struc-
tured visual input for the subsequent retrieval stage.
3.3. Retrieval-Augmented Attribute Inference
After the detection and color extraction stages, a retrieval-
augmented reasoning module is applied to infer global at-
tributes such as fabric and target gender. The module embeds
each image into a joint vision–language space using the CLIP
6

encoder. Given an imageI, its normalized embedding vectorv
is computed as
v=fCLIP(I)
∥fCLIP(I)∥ 2,(2)
wheref CLIP(·) denotes the CLIP visual encoder. For each
query embeddingv q, the similarity to all indexed catalog im-
ages{v i}in the FAISS database is obtained through cosine sim-
ilarity,
s(Iq,Ii)=vq·vi
∥vq∥∥v i∥.(3)
The Top-Kmost similar items are retrieved and used to per-
form attribute voting. Let each neighborihave an associated at-
tribute labely i(e.g., “cotton”, “female”). A similarity-weighted
score for each labelyis calculated as
score(y)=X
i:yi=yw(s i),w(s i)=eτsi,(4)
whereτis a temperature parameter controlling the exponen-
tial weighting. The final predicted attribute ˆyand its confidence
cˆyare derived as
ˆy=arg max
yscore(y),c ˆy=score(ˆy)P
y′score(y′).(5)
Ifc ˆyfalls below a defined threshold (θ attr), the attribute is
assigned as “unknown.” This retrieval-based voting stabilizes
predictions by aggregating evidence from visually similar ex-
emplars rather than relying solely on single-image inference.
In addition to attribute inference, the Top-Kretrieved samples
also provide concise textual snippets—titles or short descrip-
tions—that serve as stylistic cues for the language generation
stage.
3.4. Caption and Hashtag Generation
The final stage of the pipeline generates descriptive captions
and context-aware hashtags by prompting an LLM with struc-
tured visual evidence. For each processed image, an evidence
packEis constructed as
E={D,A,R},(6)
whereDcontains detected garments and their color descrip-
tors,Aincludes retrieved global attributes such as fabric and
target gender, andRrepresents concise caption examples re-
trieved from the catalog index. This structured representation is
converted into a textual prompt template that guides the model
to produce detailed, visually grounded sentences.
A pre-trained language modelF LLM(·) receives the prompt
containingEand generates a fluent caption ˆCand a comple-
mentary set of hashtagsHas:
{ˆC,H}=F LLM(E,P),(7)
wherePdenotes the curated prompt instructions defining
tone, format, and length. The caption emphasizes garment
attributes and visual harmony, while the hashtag set balancesgeneral and specific tags related to color, fabric, and occasion.
Since the model operates in a retrieval-augmented mode, lin-
guistic style is guided by the examples inR, ensuring descrip-
tive diversity without template repetition.
3.5. BLIP Baseline for Comparison
Fine-tuned BLIP model was utilized as a supervised base-
line to evaluate the effectiveness of the suggested retrieval-
augmented framework. BLIP combines a vision encoder with a
language decoder, which are jointly trained on image-text pairs,
which allows it to generate captions end-to-end. In this pa-
per, the model was trained on the filtered fashion dataset with
the same training-validation splits as the retrieval index. The
fine-tuning task minimized the cross-entropy loss between gen-
erated and reference captions, enabling the model to acquire
brand-specific language patterns and visual semantics. In in-
ference, the BLIP baseline made captions independently of im-
ages without extra retrieval or formatted attribute input. This
is where the difference between the traditional end-to-end cap-
tioners and the suggested modular, retrieval-enhanced model
that explicitly adds visual qualities and stylistic examples to
caption and hashtag generation lies.
3.6. Evaluation Metrics and Experimental Design
In the experimental analysis, the two important parts of the
system such as visual perception and text generation were eval-
uated using standard measures. These tests are measures of the
accuracy of the perceptual component in detecting as well as the
linguistic quality and diversity of the textual outputs produced.
3.6.1. Object Detection
The YOLO-based detector was evaluated using mean Aver-
age Precision (mAP) at IoU thresholds of 0.5 and 0.5–0.95, con-
sistent with the COCO evaluation protocol:
mAP=1
NcNcX
i=1Z1
0pi(r)dr,(8)
wherep i(r) denotes the precision–recall curve for classiand
Ncis the number of garment categories. This measure jointly
captures localization accuracy and classification precision for
multi-garment scenes.
3.6.2. Caption Quality
Retrieval-augmented pipeline and the BLIP baseline were
compared in terms of caption fluency and descriptive accuracy
measured by BLEU, METEOR, and ROUGE-L. The scores
quantify lexical and structural similarity between generated
captions and reference descriptions to evaluate the capacity of
each model to recreate garment attributes, color terms, and con-
textual relationships. Besides text measures, a CLIP similarity
measure was used to estimate visual-semantic correspondence
between images and captions. Using the CLIP ViT-B/32 en-
coder, cosine similarity was computed between each image em-
beddingv Iand its caption embeddingv T:
7

CLIPSim=vI·vT
∥vI∥∥v T∥.(9)
The mean similarity scores were reported for both generated
captions and original product descriptions, and their difference
∆ =CLIPSim pred−CLIPSim origindicates whether generated
captions exhibit stronger or weaker visual correspondence rela-
tive to the original text. This combination of lexical and seman-
tic metrics provides a balanced evaluation of linguistic quality
and visual grounding across models.
3.6.3. Hashtag Evaluation
For the RAG-LLM system, two complementary metrics were
employed to assess the quality of generated hashtags. The
first,attribute coverage, measures how effectively the predicted
hashtags capture key visual facets such as garment category,
dominant color, fabric, and target gender. For each imagei, an
attribute coverage ratio is computed as
cov i=P
f∈F ihiti(f)
|Fi|,(10)
whereF idenotes the set of known facets for imagei, and
hiti(f)=1 if any synonym of the facet value appears in the
generated hashtags, and 0 otherwise. Synonym dictionaries are
used to normalize linguistic variants such asmen/male/mensor
woman/women/female. An image is considered correctly cov-
ered if cov i≥τ, whereτis a threshold (set to 0.5 in our experi-
ments). The overall coverage is then defined as
Coverage@τ=|{i|cov i≥τ}|
Total images.(11)
The second metric,Distinct-n, quantifies linguistic diver-
sity by computing the ratio of uniquen-grams to totaln-grams
across all generated hashtags:
Distinct-1=Unique unigrams
Total unigrams,Distinct-2=Unique bigrams
Total bigrams.
(12)
Together, these measures capture both the semantic relevance
of hashtags to image content and the lexical diversity of the gen-
erated outputs, providing a comprehensive evaluation of social-
media-oriented text generation.
4. Implementation
All components of the proposed framework were developed
in Python using a modular architecture that integrates percep-
tion, retrieval, and language generation within a unified work-
flow. Experiments were conducted in a cloud-based environ-
ment on Google Colab, utilizing an NVIDIA A100 GPU. The
deep learning modules were implemented in PyTorch, with
OpenAI CLIP and FAISS used for visual embedding and simi-
larity indexing. The BLIP baseline was implemented using the
transformersanddatasetslibraries from Hugging Face.
LLM prompting was executed via the Groq API using theLLAMA 3 backbone. The entire pipeline—including YOLO-
based multi-garment detection, color extraction, retrieval, BLIP
fine-tuning, and caption generation—was executed as modu-
lar, reproducible scripts with fixed random seeds and consistent
dataset splits across all experimental runs.
4.1. Dataset Setup and Details
A total of approximately 3,000 fashion images were collected
from various social media pages and official brand websites,
covering a wide range of apparel categories, poses, and back-
grounds to capture real-world diversity. In addition, a subset of
1,200 catalog-style product images was curated with accompa-
nying metadata containing fields such as product title, textual
description, fabric type, and dominant color. The combined
dataset thus represented both consumer-generated and catalog
imagery, providing a balanced foundation for multi-garment de-
tection and captioning tasks.
All images were manually annotated in Roboflow using
bounding boxes and class labels corresponding to primary gar-
ment types such as shirt, trouser, dupatta, frock, co-ord set,
scarf, suit, shawl and jeans. The annotated data were merged
and augmented through Roboflow’s automated pipeline, result-
ing in 4,725 training images and 1,466 test images. Preprocess-
ing operations included auto-orientation correction and resiz-
ing, where all images were stretched to a uniform resolution of
640×640 pixels to standardize the detector input.
Augmentation was applied to increase data diversity and im-
prove model generalization. For each training sample, three
augmented outputs were generated using the following transfor-
mations: horizontal flips; 90° clockwise and counterclockwise
rotations; random crops with 0–20% zoom; rotations between
−15◦and+15◦; grayscale applied to 15% of images; brightness
variations within±15%; Gaussian noise applied to 0.1% of pix-
els; and bounding-box perturbations including rotation (±15°)
and blur (up to 2.5 px). These augmentations produced a visu-
ally varied and balanced dataset for robust YOLO training.
4.2. YOLO Training and Multi-Garment Inference
A YOLO-based detector was employed to identify multi-
ple garments within each fashion image. The lightweight
YOLOv11smodel from the Ultralytics framework was initial-
ized with pretrained weights (yolo11s.pt) and fine-tuned on
the curated dataset described in Section 4.1. Training was per-
formed using the officialUltralyticspackage in Python. The
network was trained for 100 epochs with an input resolution
of 640×640 px, optimizing both localization and classification
heads for all annotated apparel categories. Roboflow-generated
augmentation ensured robustness to viewpoint variation, bright-
ness shifts, and partial occlusion. Model performance was mon-
itored through validation mean Average Precision (mAP) at IoU
thresholds of 0.5 and 0.5–0.95, selecting the best checkpoint for
downstream inference.
During inference, detections were filtered using a confidence
threshold of 0.35 and a non-maximum suppression IoU thresh-
old of 0.6 to preserve overlapping garments. All bounding
8

boxes above these thresholds were retained and cropped to pro-
duce individual garment regions. These localized crops, to-
gether with class predictions, served as inputs for subsequent
color extraction and retrieval-augmented attribute inference.
4.3. Color Extraction and Attribute Processing
For each detected garment region, localized color analysis
was performed to extract the dominant visual attributes that
guide retrieval and caption generation. The cropped detec-
tions were processed through ak-means clustering algorithm
implemented withscikit-learn, where the number of clus-
ters was fixed tok=4. Pixel values were sampled uniformly
within each crop, and clustering was executed in the RGB color
space. The two largest clusters, representing the most visually
significant colors, were retained as the primary and secondary
tones. To suppress noise, near-white and near-black clusters
with coverage below 6% of the region were discarded. Each re-
maining cluster centroid was converted from RGB to CIELAB
coordinates and matched to the closest perceptual color name
in a predefined palette using Euclidean distance. The result-
ing color tokens were appended to the corresponding garment
detections, forming structured descriptors used as part of the
retrieval-augmented reasoning stage.
4.4. CLIP Embeddings and FAISS Index Construction
To support retrieval-augmented reasoning, a structured cat-
alog subset containing 1,195 product images with complete
metadata—titles, descriptions, fabric type, color, and gen-
der—was used to construct the retrieval index. The dataset was
divided into 80% training and 20% test partitions in a category-
aware manner, ensuring that each garment class contributed at
least one image to both splits. Each image was converted to
RGB format and encoded using the CLIP ViT-B/32 visual back-
bone. The normalized embeddings were indexed using FAISS,
enabling efficient inner-product similarity search equivalent to
cosine similarity on normalized vectors. Metadata for each in-
dexed image, including attribute labels and short text descrip-
tions, was stored in accompanying JSONL files for structured
access. During inference, the system queried the FAISS index
to retrieve the Top-K(K=20) most similar items, applying
similarity-weighted voting to infer global attributes (fabric and
gender) and sampling retrieved textual snippets as stylistic cues
for caption generation.
4.5. LLM Integration for Caption and Hashtag Generation
The last generation phase was introduced by a FastAPI in-
ference service that would be connected to the Groq LLM API,
based on the Llama 3 backbone. Prompt orchestration was han-
dled via thelangchain_groqinterface, defining separate tem-
plates for caption and hashtag generation. In every analyzed
image, the YOLO detections, localized color descriptors, and
retrieval-augmented features (fabric, gender) were serialised
into a structured evidence pack and injected into the caption
prompt. The instructing caption then asked the model to com-
pose a 2-3 sentence, fluent, and image-driven caption that men-
tioned colors, fabrics, and garments that had been identified.The hashtag trigger then generated 15-18 varied tags of broad,
mid-tier, and niche fashion keywords. The parameters of the
model were adjusted to a temperature of approximately 0.7 and
to a maximum length of output of approximately 250 tokens to
ensure that there was creativity and coherence. In the absence
of the LLM or API, a rule-based fallback generator generated
descriptive sentences and simple hashtags based on identified
classes, colors, and derived attributes to maintain continuous
operation when using offline execution.
4.6. BLIP Baseline Setup and Details
The BLIP baseline was fine-tuned as a supervised vi-
sion–language model to provide a comparative reference for the
proposed retrieval-augmented pipeline. This was implemented
using the “Salesforce/blip-image-captioning-base” checkpoint
from the Hugging Face library, using theAutoProcessor
andBlipForConditionalGenerationinterfaces within the
transformersframework. The same 80/20 category-balanced
data split used for the RAG index was adopted to maintain con-
sistency across experiments. Training was performed for five
epochs using the AdamW optimizer with a learning rate 5×10−5
and mixed-precision computation on an NVIDIA A100 GPU.
Each batch included paired image–caption samples processed
into input token IDs and pixel embeddings. The model was
optimized via cross-entropy loss between generated and refer-
ence captions, updating both visual and textual parameters end-
to-end. The loss curves were tracked at both batch and epoch
levels to verify the stability of the convergence, and the check-
point with the lowest validation loss had been selected. In the
inference phase, the fine-tuned BLIP model produced captions
without retrieval or attribute conditioning using images as di-
rect inputs.
The entire implementation comprises all modules in end-to-
end pipeline. YOLO detection, color extraction, CLIP retrieval
and LLM generation are sequentially run with unified settings,
whereas the BLIP baseline is a supervised baseline. All stages
were applied on modular scripts with fixed parameters and con-
stant dataset splits making them comparable. This unified ar-
rangement offers a firm basis on which the accuracy of percep-
tion and captioning performance could be assessed in the latter
results section.
5. Results
5.1. YOLO Detection Performance
The YOLO-based garment detector was evaluated on 1,466
test images containing 1,974 annotated instances across nine
apparel categories. Figure 2 presents the per-class detection
performance in terms of mAP@0.5 and mAP@0.5:0.95. The
overall mAP@0.5 reached 0.709, confirming reliable local-
ization and classification of multiple garments within com-
plex social and catalog scenes. Among individual categories,
jeansandkurtaachieved the highest detection accuracy with
mAP@0.5 scores of 0.85 and 0.99 respectively, followed by
frock(0.80) andshirt(0.67). Relatively lower scores fordu-
pattaandtrouserwere attributed to fine-grained boundaries and
9

Figure 2: Per-class detection performance of YOLO showing mAP@0.5 and
mAP@0.5:0.95 across garment categories.
Figure 3: Example detection outputs illustrating accurate multi-garment recog-
nition in social and catalog scenes.
occlusion, which occasionally caused partial detections or mis-
classifications. The corresponding mAP@0.5:0.95 values show
a consistent decline across categories, reflecting the expected
drop in performance under stricter localization thresholds.
Figure 3 illustrates representative qualitative results demon-
strating the model’s ability to detect multiple garments in a sin-
gle frame. The detector successfully identifies overlapping ap-
parel such as shirts, dupattas, and trousers while preserving
their spatial arrangement. The results confirm that the fine-
tuned YOLOv11s model generalizes effectively across light-
ing variations, pose diversity, and complex social backgrounds,
providing a robust foundation for the subsequent captioning and
retrieval modules.
5.2. Caption Quality: BLIP vs RAG-LLM
Caption generation results were evaluated for both the BLIP
baseline and the proposed retrieval-augmented pipeline using
BLEU, METEOR, ROUGE, and CLIP similarity metrics. Ta-
ble 2 summarizes the quantitative performance across mod-
els. The fine-tuning process for BLIP converged smoothly,
as illustrated in Fig. 4, where both batch-level and epoch-
level losses steadily decreased across five epochs, indicating
stable optimization without overfitting. The fine-tuned BLIP
model achieved higher BLEU (0.2120), METEOR (0.5845),
Figure 4: BLIP fine-tuning loss curves showing batch-level and epoch-level
training loss over five epochs.
and ROUGE-L (0.4194) scores, indicating strong lexical align-
ment with the reference descriptions. These results reflect
BLIP’s supervised learning behavior—memorizing the linguis-
tic patterns present in the training captions—leading to syn-
tactically accurate but templated outputs with limited stylistic
variation. Because BLIP is fine-tuned for a specific dataset, ex-
panding it to new clothing categories would require additional
labeled data and retraining to maintain quality, limiting its scal-
ability across broader fashion domains.
In contrast, the RAG-LLM pipeline scored lower on n-gram-
based scores (BLEU=0.0230, METEOR=0.1374, ROUGE-
L=0.1340) because its captions are not optimized around
word-level overlap. It rather composes accounts based on re-
covered evidence of attributes and on stylistic exemplification,
and produces fluent, human-like, and context-sensitive narra-
tives based on the factual garment characteristics like color,
fabric, and category. This formulation, which is founded on
retrieval, enables the pipeline to extrapolate on missing or low-
resource categories of fashions without further fine-tuning. The
qualitative comparison in Fig. 5 shows that the qualitative cap-
tions of RAG-LLM are more contextualized and descriptively
realistic than those of BLIP.
The CLIP similarity analysis provides complementary in-
sight into semantic grounding. For BLIP, the mean simi-
larity between image and predicted caption (0.3134) slightly
exceeded that of the original product descriptions (0.3098),
indicating close visual–semantic alignment but minimal lin-
guistic novelty. RAG-LLM captions, while showing a lower
similarity (0.2827) relative to the originals (0.3102), main-
tain factual grounding while achieving greater linguistic di-
versity. This trade-offis desirable for creative fashion narra-
tives, where human-style expression is prioritized over exact
textual replication. Overall, BLIP excels in lexical fidelity for
known classes, whereas RAG-LLM delivers more generaliz-
able, attribute-driven, and hallucination-resistant captions suit-
able for large-scale automated fashion content generation.
5.3. Hashtag Evaluation
The generated hashtags from the RAG-LLM pipeline were
evaluated using the attribute coverage and diversity metrics de-
scribed in Section 3.6. Quantitative results are summarized
in Table 3. The system achieved exceptionally high cover-
age across all evaluated thresholds, with a mean attribute cov-
erage of 0.7976 and Coverage@0.5 equal to 1.000, indicat-
10

Table 2: Caption quality comparison between BLIP and RAG-LLM using BLEU, METEOR, ROUGE (F-scores), and CLIP similarity.
Model BLEU METEOR R1-F R2-F RL-F CLIP pred
BLIP 0.2120 0.5845 0.4336 0.3058 0.4194 0.3134
RAG-LLM 0.0230 0.1374 0.1556 0.0358 0.1340 0.2827
Original Caption:“Chic and sophisticated, our Velvet Dyed Co-Ord Set in a
deep blue hue features a velvet shirt paired with straight pants. This solid suit
is perfect for making a stylish statement at any event.”
BLIP Generated Caption:“multicolorechic and sophisticated, our velvet
dyed co - ord set in a deep blue hue features a velvet shirt paired with straight
pants. this elegant suit is perfect for making a stylish statement at any event.”
RAG-LLM Generated Caption:“Elevate your evening style with our velvet
co-ord set, featuring a black velvet shirt and straight pants that exude
sophistication and comfort, perfect for a formal winter event.”
RAG-LLM Generated Hashtags:“#FashionForWomen #VelvetClothing
#WinterFashion #FormalWear #BlackOutfit #CoOrdSet #LuxuryFashion
#EveningStyle #VelvetShirt #StraightPants #FemaleFashion
#SophisticatedStyle #WinterFormalWear #VelvetFashionTrends
#BlackVelvetOutfit #Women”
Figure 5: Qualitative caption comparison for a sample test image showing the
original brand description, BLIP baseline output, and RAG-LLM generated
caption and hashtags.
Figure 6: Comparison of caption quality metrics (BLEU, METEOR, ROUGE-
L, CLIP similarity) for BLIP and RAG-LLM.
ing that every test image contained hashtags reflecting at least
half of its key visual facets (garment category, dominant color,
fabric, and gender). Even under stricter thresholds, Cover-
age@0.6 and Coverage@0.7 remained above 0.98, confirming
that nearly all outputs correctly captured 60–70% of their corre-
sponding attributes. These results highlight the factual ground-
ing and attribute consistency of the retrieval-augmented gener-
ation framework, where the structured evidence pack ensures
that generated hashtags accurately represent the visual content.
Distinct-nratios were used to measure the linguistic diver-
sity of the produced hashtags. The score of 0.0401 in Distinct-1
and 0.3188 in Distinct-2 indicate a moderate level of the lexical
variation at the unigram level but high bigram diversity, indi-
cating that individual tokens (e.g., fashion, style) are repeated
across pictures but their combinations create context-rich and
unique expressions. Such a high attribute coverage and moder-
ate linguistic diversity show that the RAG-LLM paradigm pro-
duces hashtags that are semantically faithful but stylistically di-
verse and can be applied to the real-life fashion marketing and
social media use.
6. Discussion and Future Work
The overall results of the experiment prove that the suggested
retrieval-augmented captioning pipeline is effective in balanc-
ing between structured attribute reasoning and human-like nar-
rative generation. All of these components, object detection,
attribute retrieval, and language generation, are involved in cre-
ating visually faithful and stylistically fluent results that can be
used in the real world in fashion segment. The YOLO-based de-
tector was shown to be highly accurate among various garment
classes and also capable of detecting various apparel objects
with high reliability even when used in social media images
with a complex background. High detection consistency was
a direct support of downstream captioning and hashtag steps,
whereby each crop and its properties gave factual support to
11

Table 3: Quantitative results for hashtag generation showing attribute coverage and diversity metrics.
Total Images Mean Coverage Coverage@0.5 Coverage@0.6 Coverage@0.7 Distinct-1 Distinct-2
226 0.7976 1.000 0.9823 0.9823 0.0401 0.3188
text generation.
The BLIP baseline versus the RAG-LLM pipeline compari-
son shows that there is a definite trade-offbetween the lexical
fidelity and the contextual expressiveness. Trained in a super-
vised way on paired brand descriptions, BLIP scored high on
BLEU, METEOR, and ROUGE, which means that it is related
closely to the reference corpus. Nevertheless, such captions are
likely to mimic the syntactic patterns of the fine-tuning process,
which leads to the repetitive syntactic patterns and the lack of
stylistic diversity. Since BLIP relies on a fixed dataset, it would
need to introduce more labeled text-image pairs and retrain to
be applied to new clothing categories or new regions of fashion,
which limits its flexibility.
Conversely, the retrieval-augmented approach extrapolates
beyond the particular categories observed in the construction
of the index. The RAG-LLM system builds descriptions based
on visual detection (category, color) and retrieval-based infer-
ence (fabric, gender) attributes, which make them factually ac-
curate on previously unseen garments. The qualitative results
indicate that RAG-LLM captions are more likely to be more
marketing-like in style with a focus on texture, coordination,
and aesthetic context instead of direct reproduction of catalog
language. Even though its n-gram overlap with reference text is
less, captions are characterized by high CLIP similarity, which
proves that visual-semantic coherence is maintained. Such be-
havior is indicative of the fact that retrieval conditioning is use-
ful in reducing hallucination and improving factual consistency,
which is a desired attribute of fashion e-commerce and content
creation pipelines.
The hashtag evaluation further validates these strengths. The
proposed attribute-coverage metric shows that nearly all gen-
erated hashtags represent at least 60–70% of the visual facets,
while the moderate Distinct-nscores confirm balanced diversity
without uncontrolled randomness. These results highlight that
the model not only captures the key product attributes but also
presents them through varied and socially relevant language
patterns, supporting engagement-oriented applications such as
automated post generation or trend analysis.
Even though it has its benefits, the proposed pipeline has
some drawbacks. The quality of the indexed catalog and its rep-
resentativeness are very important in the retrieval stage; noise
and biased metadata may be transferred to the generated cap-
tions. On the same note, the detector can be generalized well
in the domain of South Asian fashion, but it might need retrain-
ing or inclusion of additional classes to address other types of
garments or accessories. The existing CLIP and LLM mod-
ules can also be used separately; more intimate multimodal in-
teraction would provide additional improvements to visual-text
alignment. In addition, the current system only accepts English
captions, which restricts its use in multilingual fashion markets.
The future work will be oriented to three main directions.To begin with, the retrieval index should be expanded with
bigger and more varied brand sets to enhance the robustness
of attributes and their stylistic diversity. Second, a wider re-
gional adoption could be made possible by incorporating mul-
tilingual caption generation through cross-lingual retrieval and
LLM prompting. Third, the CLIP encoder and the captioning
model can be fine-tuned together in a single retrieval-generation
loop, which could enhance semantic cohesion and decrease re-
liance on fixed metadata. Lastly, user engagement metrics (like
click-through or like rates) would offer a viable authentication
of caption and hashtag efficiency in social settings.
7. Conclusion
This work introduced a retrieval-augmented system of fash-
ion caption and hashtag generation that combines multiple gar-
ment detection, visual attribute justification, and LLM prompt-
ing. The system exhibited high detection, factual grounding,
and stylistic fluency in multifaceted fashion pictures. The RAG-
LLM pipeline generated more human-like, attribute-centric,
and generalizable captions than the fine-tuned BLIP baseline
and was semantically aligned with visual data. The reliability
of generated hashtags was also confirmed by the proposed at-
tribute coverage and diversity measure. Collectively, these find-
ings demonstrate the opportunity of retrieval-enhanced genera-
tion as a scalable remedy to visually grounded, socially versa-
tile content creation in fashion. Further extensions to multilin-
gual captioning, larger retrieval corpora and unified multimodal
optimization can improve its performance and practical use in
real-world implementation.
Data Availability
The data used for this research will be made available upon
request.
Funding
This research did not receive any specific grant from funding
agencies in the public, commercial, or not-for-profit sectors.
References
[1] Y . A. Thakare, K. H. Walse, A review of deep learning im-
age captioning approaches, Journal of Integrated Science
& Technology 12 (1) (2024) 712.
[2] L. Xu, Q. Tang, J. Lv, B. Zheng, X. Zeng, W. Li, Deep
image captioning: A review of methods, trends and future
challenges, Neurocomputing 546 (2023) 126287.
12

[3] J. Jia, X. Ding, S. Pang, X. Gao, X. Xin, R. Hu, J. Nie,
Image captioning based on scene graphs: A survey, Expert
Systems with Applications 231 (2023) 120698.
[4] R. Gusain, S. Pathak, S. Ghosh, Instagram post cap-
tion recommendation engine, in: Proceedings of the 14th
International Conference on Computing Communication
and Networking Technologies (ICCCNT), IEEE, 2023,
pp. 1–3.
[5] M. Jafari Sadr, S. L. Mirtaheri, S. Greco, K. Borna,
Popular tag recommendation by neural network in so-
cial media, Computational Intelligence and Neuroscience
2023 (1) (2023) 4300408.
[6] E. Balloni, R. Pietrini, M. Fabiani, E. Frontoni,
A. Mancini, M. Paolanti, Social4fashion: An intelligent
expert system for forecasting fashion trends from social
media contents, Expert Systems with Applications 252
(2024) 124018.
[7] S. M. Islam, S. Joardar, A. A. Sekh, A survey on fashion
image retrieval, ACM Computing Surveys 56 (6) (2024)
1–25.
[8] Y . Tang, L. Zhang, Y . Yuan, Z. Chen, Improving fash-
ion captioning via attribute-based alignment and multi-
level language model, Applied Intelligence 53 (24) (2023)
30803–30821.
[9] P. Mahalakshmi, N. S. Fatima, Summarization of text
and image captioning in information retrieval using deep
learning techniques, IEEE Access 10 (2022) 18289–
18297.
[10] G. Lavanya, S. D. Pande, Enhancing real-time object de-
tection with yolo algorithm, EAI Endorsed Transactions
on Internet of Things 10 (2023).
[11] C. H. Kang, S. Y . Kim, Real-time object detection and
segmentation technology: An analysis of the yolo algo-
rithm, JMST Advances 5 (2) (2023) 69–76.
[12] W. He, J. Wang, L. Wang, R. Pan, W. Gao, A seman-
tic segmentation algorithm for fashion images based on
modified mask rcnn, Multimedia Tools and Applications
82 (18) (2023) 28427–28444.
[13] Y .-H. Chang, Y .-Y . Zhang, Deep learning for clothing
style recognition using yolov5, Micromachines 13 (10)
(2022) 1678.
[14] U. Sirisha, B. S. Chandana, Semantic interdisciplinary
evaluation of image captioning models, Cogent Engineer-
ing 9 (1) (2022) 2104333.
[15] B. Yu, Z. Zhong, X. Qin, J. Yao, Y . Wang, P. He, Auto-
mated testing of image captioning systems, in: Proceed-
ings of the 31st ACM SIGSOFT International Symposium
on Software Testing and Analysis, 2022, pp. 467–479.[16] T. Ghandi, H. Pourreza, H. Mahyar, Deep learning ap-
proaches on image captioning: A review, ACM Comput-
ing Surveys 56 (3) (2023).
[17] I. Khurram, M. M. Fraz, M. Shahzad, N. M. Rajpoot,
Dense-captionnet: A sentence generation architecture for
fine-grained description of image semantics, Cognitive
Computation 13 (2021) 595–611.
[18] A. M. Rinaldi, C. Russo, C. Tommasino, Automatic image
captioning combining natural language processing and
deep neural networks, Results in Engineering 18 (2023)
101107.
[19] A. Nursikuwagus, R. Munir, M. L. Khodra, Hybrid of
deep learning and word embedding in generating cap-
tions: Image-captioning solution for geological rock im-
ages, Journal of Imaging 8 (11) (2022) 294.
[20] S. Li, M. Dang, Y . Xu, A. Wang, Y . Guo, Bridge damage
description using adaptive attention-based image caption-
ing, Automation in Construction 165 (2024) 105525.
[21] R. M. Elsady, Y . A. Ahmed, M. A.-M. Salem, Remote
sensing image segmentation and captioning using deep
learning, in: Proceedings of the 2nd International Con-
ference on Smart Cities 4.0, IEEE, 2023, pp. 196–201.
[22] H. Han, B. O. Aboubakar, M. Bhatti, B. A. Talpur, Y . A.
Ali, M. Al-Razgan, Y . Y . Ghadi, Optimizing image cap-
tioning: The effectiveness of vision transformers and
vgg networks for remote sensing, Big Data Research 37
(2024) 100477.
[23] D. He, C. Xie, Semantic image segmentation algorithm in
a deep learning computer network, Multimedia Systems
28 (6) (2022) 2065–2077.
[24] I. Tereikovskyi, Z. Hu, D. Chernyshev, L. Tereikovska,
O. Korystin, O. Tereikovskyi, The method of semantic
image segmentation using neural networks, International
Journal of Image, Graphics and Signal Processing 10 (6)
(2022) 1.
[25] Y . Wang, B. Xiao, A. Bouferguene, M. Al-Hussein, H. Li,
Vision-based method for semantic information extraction
in construction by integrating deep learning object detec-
tion and image captioning, Advanced Engineering Infor-
matics 53 (2022) 101699.
[26] M. Al Duhayyim, S. Alazwari, H. A. Mengash, R. Mar-
zouk, J. S. Alzahrani, H. Mahgoub, F. Althukair, A. S.
Salama, Metaheuristics optimization with deep learning
enabled automated image captioning system, Applied Sci-
ences 12 (15) (2022) 7724.
[27] J. Choi, Y . Kwon, I. Kim, Item-region-based style clas-
sification network (irsn): A fashion style classifier based
on domain knowledge of fashion experts, Applied Intelli-
gence 54 (20) (2024) 9579–9593.
13

[28] A. Kalinin, A. A. Jafari, E. Avots, C. Ozcinar, G. An-
barjafari, Generative ai-based style recommendation us-
ing fashion item detection and classification, Signal, Im-
age and Video Processing 18 (2024) 9179–9189.
[29] C. Xu, H. Tan, J. Li, P. Li, Understanding social me-
dia cross-modality discourse in linguistic space, arXiv
preprint arXiv:2302.13311 (2023).
14