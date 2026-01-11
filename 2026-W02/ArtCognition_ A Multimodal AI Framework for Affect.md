# ArtCognition: A Multimodal AI Framework for Affective State Sensing from Visual and Kinematic Drawing Cues

**Authors**: Behrad Binaei-Haghighi, Nafiseh Sadat Sajadi, Mehrad Liviyan, Reyhane Akhavan Kharazi, Fatemeh Amirkhani, Behnam Bahrak

**Published**: 2026-01-07 17:35:37

**PDF URL**: [https://arxiv.org/pdf/2601.04297v1](https://arxiv.org/pdf/2601.04297v1)

## Abstract
The objective assessment of human affective and psychological states presents a significant challenge, particularly through non-verbal channels. This paper introduces digital drawing as a rich and underexplored modality for affective sensing. We present a novel multimodal framework, named ArtCognition, for the automated analysis of the House-Tree-Person (HTP) test, a widely used psychological instrument. ArtCognition uniquely fuses two distinct data streams: static visual features from the final artwork, captured by computer vision models, and dynamic behavioral kinematic cues derived from the drawing process itself, such as stroke speed, pauses, and smoothness. To bridge the gap between low-level features and high-level psychological interpretation, we employ a Retrieval-Augmented Generation (RAG) architecture. This grounds the analysis in established psychological knowledge, enhancing explainability and reducing the potential for model hallucination. Our results demonstrate that the fusion of visual and behavioral kinematic cues provides a more nuanced assessment than either modality alone. We show significant correlations between the extracted multimodal features and standardized psychological metrics, validating the framework's potential as a scalable tool to support clinicians. This work contributes a new methodology for non-intrusive affective state assessment and opens new avenues for technology-assisted mental healthcare.

## Full Text


<!-- PDF content starts -->

1
ArtCognition: A Multimodal AI Framework
for Affective State Sensing from Visual
and Kinematic Drawing Cues
Behrad Binaei-Haghighi, Nafiseh Sadat Sajadi, Mehrad Liviyan, Reyhane Akhavan Kharazi,
Fatemeh Amirkhani, and Behnam Bahrak
Abstract—The objective assessment of human affective and
psychological states presents a significant challenge, particularly
through non-verbal channels. This paper introduces digital draw-
ing as a rich and underexplored modality for affective sensing.
We present a novel multimodal framework, named ArtCognition,
for the automated analysis of the House-Tree-Person (HTP) test,
a widely used psychological instrument. ArtCognition uniquely
fuses two distinct data streams: static visual features from the
final artwork, captured by computer vision models, and dynamic
behavioral kinematic cues derived from the drawing process
itself, such as stroke speed, pauses, and smoothness. To bridge
the gap between low-level features and high-level psychological
interpretation, we employ a Retrieval-Augmented Generation
(RAG) architecture. This grounds the analysis in established
psychological knowledge, enhancing explainability and reducing
the potential for model hallucination. Our results demonstrate
that the fusion of visual and behavioral kinematic cues provides
a more nuanced assessment than either modality alone. We
show significant correlations between the extracted multimodal
features and standardized psychological metrics, validating the
framework’s potential as a scalable tool to support clinicians. This
work contributes a new methodology for non-intrusive affective
state assessment and opens new avenues for technology-assisted
mental healthcare.
Index Terms—Object Detection, Multimodal Learning, Large
Language Model, Retrieval-Augmented Generation, Digital
Drawing Analysis, Psychological Assessment
I. INTRODUCTION
MENTAL health issues affect a substantial portion of the
global population and impose significant personal and
societal costs. For instance, in 2019, it was estimated that “one
in every eight, or 970 million people in the world, lives with
a mental disorder” [1]. Despite this high prevalence, accurate
diagnosis remains challenging. Current clinical tools, such as
the Diagnostic and Statistical Manual of Mental Disorders
(DSM) and the International Classification of Diseases (ICD),
have limitations, including overlapping criteria, binary catego-
rizations, and neglect of contextual or behavioral factors [2].
This diagnostic gap is particularly problematic in contexts with
Behrad Binaei-Haghighi and Nafiseh Sadat Sajadi contributed equally to
this work; Nafiseh Sadat Sajadi and Behnam Bahrak are cocorresponding
authors.
Behrad Binaei-Haghighi, Mehrad Liviyan, and Reyhane Akhavan Kharazi
are with the Department of Electrical and Computer Engineering, University
of Tehran, Tehran, Iran.
Nafiseh Sadat Sajadi and Behnam Bahrak are with the Tehran Institute for
Advanced Studies, Khatam University, Tehran, Iran.
Fatemeh Amirkhani is with the Department of Psychology, Allameh
Tabataba’i University, Tehran, Iran.limited mental health resources, where the lack of specialists
delays timely detection and treatment [3]. Consequently, there
is a growing need for innovative tools that can complement
conventional diagnostic practices by providing accessible, ob-
jective, and scalable assessment methods [4].
Among various clinical assessment techniques, projective
drawing tests have a long history of providing insights into
human emotions and attitudes, as well as revealing under-
lying psychodynamics [5]. The House–Tree–Person drawing
test, first proposed by Buck in 1948, remains one of the
most widely used projective measures, ranking eighth among
commonly applied psychological assessments in an Ameri-
can Psychological Association survey [6], [7]. The HTP test
offers several advantages, including spontaneity, structural
complexity, a non-verbal mode of expression, and cultural
independence, which collectively enhance its diagnostic utility
[8], [9].
However, traditional drawing assessments face both practi-
cal and methodological challenges. Manual scoring is time-
consuming and often fails to capture important process-level
cues such as the order of drawing objects, number of actions
for drawing an object, pauses, and erasing behavior. At the
same time, psychology is undergoing rapid digitalization, with
an expanding ecosystem of digital tools for mental healthcare
[10], enabling AI-driven transformation of projective tests and
their analysis. Recent advances in computer vision and deep
learning now make it feasible to analyze drawings at scale
with greater objectivity and consistency.
Building on these developments, we propose a framework
that integrates a digital drawing web app with an AI-assisted
interpretation platform to make projective drawing assessments
more accessible. The system automatically logs drawing ac-
tions, capturing process-level metadata without requiring con-
tinuous expert supervision. Moreover, computer vision models
detect and classify key components, which are organized into
clinically relevant categories using a rule-based psychoanalysis
metrics. Subsequently, a description generator synthesizes the
results from the computer vision models and the process-level
metadata from the input to create a comprehensive textual
description of the drawing. This description is used as a query
for a retrieval-augmented generation module which produces
explanations grounded in organized psychological knowledge.
As a result, using relevant data chunks helps reducing halluci-
nation with combining behavioral logging, object-level analy-
sis, and knowledge-guided reasoning. ArtCognition advancesarXiv:2601.04297v1  [cs.LG]  7 Jan 2026

2
HTP assessment toward a standardized, scalable, and user-
friendly workflow.
II. RELATEDWORKS
For interpreting the HTP test, John Buck relied on a
combination of structured scoring systems and examiner ob-
servations of the drawing process to derive clinical inferences
[6], although conventional scoring protocols are time-intensive
and heavily dependent on expert judgment. These limitations
motivate automated frameworks to support scalable assessment
[11].
Previous research suggests that partial automation of HTP
analysis has focused on object detection and rule-based feature
extraction. One-stage detectors, such as the YOLO family,
provide reliable bounding-box and instance predictions for
isolating HTP elements [12]. Recent architectures, including
EfficientNet, Vision Transformer (ViT), ConvNeXt, and Swin
Transformer, further improve accuracy for object analysis
[13]–[16]. Lee et al. [17] generated psychological analysis
tables from detected elements by computing features such
as object proportions and spatial placement, demonstrating
potential for more objective and efficient assessment. How-
ever, these approaches rely on handcrafted rules and interpret
elements in isolation, leaving final report generation to human
experts.
Subsequent studies have investigated the use of children’s
drawings for emotion classification. Alshahrani [18] used
YOLOv8n-cls to classify drawings into four emotional states
(happiness, sadness, anxiety, and anger/aggression), achieving
94% top-1 accuracy with a compact, mobile-friendly architec-
ture. However, this method is restricted to final drawing and
ignores temporal features that could provide richer interpretive
signals.
Recent advances in large language models (LLMs), such as
GPT-4, have demonstrated strong capabilities in performing
tasks that require human-level reasoning, including cogni-
tive psychology challenges. Studies have evaluated GPT-4
on established cognitive datasets such as CommonsenseQA,
SuperGLUE, MATH, and HANS, showing high accuracy
relative to prior state-of-the-art models. These results suggest
that LLMs can integrate contextual information and simu-
late aspects of human cognitive processes, highlighting their
potential to support automated psychological assessment and
interpretation [19].
Two key gaps remain in automated HTP analysis. First, most
existing systems focus only on the final drawing, ignoring
behavioral metadata which contain valuable information [11].
Second, prior approaches often separate recognition from
interpretation, with few frameworks integrating object-level
detection, behavioral analysis, and psychometric classification
in a unified, interpretable pipeline. Addressing these gaps
requires end-to-end multimodal models that combine visual,
temporal, and contextual features to enable scalable, data-
driven, and clinically meaningful psychological assessment.
III. METHODOLOGY
This study addresses the gaps in prior works with a multi-
stage approach that integrates image analysis and temporalbehavioral data analysis to enable interpretable HTP assess-
ment. First, we construct a dataset of high-resolution drawings
paired with fine-grained stroke sequences, capturing both
static visual features and behavioral information. Second, we
employ a two-stage object detection model to first localize
the main house, tree, and person components, followed by a
secondary detection stage to identify specific constituent parts
within those objects. Third, these detected components are
processed through classification models that analyze specific
psychological markers, such as a ”poker face” or a leaning
house. Fourth, we extract behavioral data from the input
metadata to track behavioral patterns like eraser usage and
stroke actions. Fifth, a description generator synthesizes the
object detection results, classification results and metadata into
a comprehensive textual summary of the drawing. Finally, this
text serves as a query for a retrieval-augmented generation
module, which produces final interpretations grounded in
expert psychological knowledge. The proposed architecture is
illustrated in Figure 1.
A. Dataset
To construct the dataset, we use digital drawing to capture
fine-grained metadata throughout the drawing process. We
collected 146 samples from volunteer participants, each using
a custom web-based drawing application designed to record
detailed user interactions. The application logs every drawing
action with high temporal and spatial resolution, enabling near-
exact reconstruction of the drawing process. For each sample,
the metadata is stored in JSON format and paired with the
final image in PNG format [20].
Each dataset sample consists of the completed drawing
along with its corresponding metadata. The metadata records
all drawing actions and includes attributes such as drawing
order, action type, color, opacity, timestamp, line width, and
a sequence of points describing each stroke trajectory. An
example of a recorded drawing action is shown in Figure 2.
In addition, 21 participants completed a supplementary
House–Tree–Person questionnaire (Appendix B), developed in
collaboration with a domain expert, to provide complementary
self-report information and improve the interpretability of the
drawings.
B. Object Detection
The object detection phase uses a two-stage hierarchical
approach involving four specialized models based on the
YOLOv11 architecture. In the first stage, a main component
detector localizes the primary HTP elements: the house, tree,
and person. It also identifies environmental features such as
clouds, sun, and mountains.
In the second stage, three individual constituent parts de-
tectors, also utilizing YOLOv11, process the cropped images
of the house, tree, and person. These models capture fine-
grained anatomical and structural details, such as windows
and doors in the house, trunks and branches in the tree,
and facial features and limbs in the person. This granular
detection ensures that all relevant sub-components are isolated
for further analysis. These detections directly inform the

3
Fig. 1: Workflow of ArtCognition
(a) Drawing Artwork
[{
"order": 1,
"action_type": "drawLine",
"color": "#000000",
"opacity": 1,
"line_width": 5,
"timestamp_start": 1751293539626,
"timestamp_end": 1751293540253,
"points": [{
"x": 333.95,
"y": 102.76,
"pointerType": "mouse",
"timestamp": 1751293539626
}]
}, ...]
(b) Sample of Drawing Metadata
Fig. 2: Digital drawing sample and its corresponding metadata.
Description Generation module by confirming the presence
or omission of specific elements. Furthermore, the resulting
bounding boxes allow for precise spatial measurements that
hold significant psychological insights. For instance, housesize is classified astiny,normal, orhuge, corresponding to
area ratios below1
9, between1
9and2
3, and above2
3of the
drawing, respectively [21].
C. Object Classification
Once the components and their parts are detected, they
are processed through six specialized classification models
to identify psychological markers. These models are selected
from high-performing architectures, including EfficientNet,
ViT, ResNet50, MobileNetV2, ConvNeXt, and Swin Trans-
former, to ensure high accuracy [22], [23].
House Classifiers Two models analyze the structural char-
acteristics of the house:
•Leaning House Classifier: Identifies if the house is tilted,
which may indicate structural or emotional instability.
•2D/3D House Classifier: Determines the perspective of
the drawing to assess the level of spatial complexity.
Tree Classifiers Two models evaluate the vitality and shape
of the tree:
•Dead Tree Classifier: Distinguishes between living trees
and those depicted as dead or withered.
•Flattened Crown Classifier: Detects deformations in the
tree’s crown, such as flattening, which serves as a clinical
indicator [11].
Person Classifiers The final two models analyze the facial
and bodily representation of the person:
•Poker Face Classifier: Evaluates the facial expression to
detect emotional neutrality or a lack of affect [11].
•Single Line Limbs Classifier: Identifies whether the limbs
are drawn as simple lines, potentially signaling emotional
or developmental constraints [24].

4
Training Process:All models were trained independently.
YOLOv11 was first trained on 117 labeled samples to localize
main objects in HTP drawings. Detected objects were then
cropped to generate training data for the subsequent classifiers.
Once trained, all models were integrated into the pipeline for
end-to-end detection and analysis of drawing details.
D. Metadata Analysis
As illustrated in Figure 2, our system captures detailed
metadata for each drawing action, including the action type
(e.g., drawing a line, using an eraser, using a bucket) and
per-point attributes such as coordinates and timing recorded
at millisecond resolution, which allow for a precise recon-
struction of the drawing [20]. More importantly, this data sup-
ports advanced analysis that provides quantitative behavioral
insights, which are often difficult or impossible for a human
or psychologist to measure manually.
Stroke Speed:The Euclidean distance between consecutive
points is used to calculate the total stroke path length. This
fine-grained measurement allows for the precise calculation
of drawing speed for individual strokes and for each detected
object (e.g., house, tree, person). The total stroke length
Lis determined by summing the distancesd ibetween all
consecutive point pairs(x i, yi)and(x i+1, yi+1). The average
stroke speedvis then computed by dividing the total stroke
length by the stroke durationT. The speed is measured in
pixels per second.
di=p
(xi+1−xi)2+ (y i+1−yi)2,(1)
L=n−1X
i=1di (2)
v=L
T(3)
Drawing speed reflects the ability to control motor, psy-
chomotor, and automatic movements [25]. In addition, another
study concluded that kinematic parameters such as speed and
changes in speed are weaker in children with Developmental
Coordination Disorder (DCD) [26].
Inter-stroke Interval:The inter-stroke interval is the time
gap between the end of one stroke and the start of the next,
extracted directly from timestamped drawing metadata. We
compute the total duration of the pause along with statistics
such as mean, variance, and distribution across the draw-
ing. These pauses between strokes can be an indicator of
cognitive-motor coordination, processing slowness, hesitation,
and shifting focus. Moreover, reduced speed was observed
particularly in the Lewy Body Dementia (LBD) group, while
increased pauses and total durations were observed in both
the Alzheimer’s Disease and LBD groups [27]. ArtCognition
precisely tracks the pauses and speed of the user while
drawing, which can enable further research into the behavioral
characteristics of specific groups. The distribution of these
time gaps and their median value are displayed in the Figure
3.
Fig. 3: Inter-stroke interval throughout the drawing process.
Stroke Smoothness:Spectral Arc Length (SPARC) is a
common metric used to quantify the smoothness of a line
by calculating the arc length of the line’s normalized Fourier
magnitude spectrum. The calculation integrates the normalized
spectrum up to an adaptive frequency cutoff (ω c), whereV(ω)
is the Fourier magnitude spectrum of the line’s first derivative
(rate of change in pressure) and ˆV(ω)is the spectrum nor-
malized by its value at zero frequency,V(0). The spectral arc
length (SAL) is then defined as:
SAL≜−Rωc
0
1
ωc2
+
dˆV(ω)
dω21
2
dω
ˆV(ω) =V(ω)
V(0)(4)
This methodology, where a value closer to zero indicates
greater smoothness, has been applied to analyze kinematic
signals and, notably, to quantify the smoothness of pharyngeal
high-resolution manometric curves in swallowing studies [28].
Placement of Drawn Objects:Neuro-psychological studies
show placement patterns change with age and with spatial
processing differences [29]. For example, children tend to
draw slightly left of center, with right-handed individuals
showing a stronger leftward bias [30], and similar tendencies
appear across cultures. For example, in river basin drawings,
children frequently depict rivers flowing from left to right or
downwards [31].
To quantify placement, we divide the canvas into a3×3
grid, capturing both overall drawing position and individual
object locations [32]. Each stroke consists of multiple points,
and we compute the normalized distribution of points across
grid cells:
Pij=nijP3
k=1P3
l=1nkl, i, j∈ {1,2,3}(5)
wheren ijis the number of points in cell(i, j). This yields
a probability distributionP ij, allowing comparison across
drawings of varying size and density.
Figure 4 visualizes the distribution of points across the nine
regions, providing a quantitative representation of drawing
placement patterns.
Eraser Usage Pattern:High frequency of erasing indicates
anxiety, self-doubt, and perfectionism [33]. On the other hand,
a 2020 study compared erasing behaviors in physical and
digital drawing environments, finding that erasing occurs more
frequently in digital settings, likely due to convenient ”undo”
functions [34]. These differences did not significantly affect
the interpretability of final drawings, suggesting that eraser

5
Fig. 4: Distribution of drawing points over the3×3canvas
grid.
use reflects compensatory behaviors rather than diagnostic
markers.
In ArtCognition, eraser behavior is analyzed along multiple
dimensions. Using bounding boxes from the object detection
model, we count erasing events per object to capture user
sensitivity to specific elements. We also compute total erasing
time and cumulative erased area, providing a behavioral profile
of revision strategies and self-monitoring during the drawing
process. Generally, redrawn pencil lines evoke a sense of self-
criticality and self-doubt [35]. As a result, our module is ca-
pable of providing insights into the user’s eraser usage patterns
by using metadata. Moreover, with the help of bounding boxes
detected by YOLO, our module can determine how many times
a user uses the eraser to modify each object.
Order of Drawing:In the HTP test, the sequence in which
the examinee chooses to draw the house, tree, or person is
considered a behavioral indicator reflecting the individual’s
spontaneous focus of attention and emotional priorities [36].
Moreover, drawing the person first is often associated with
heightened self-focus, body-image concerns, or interpersonal
sensitivity, particularly when accompanied by detailed correc-
tions or hesitation. Drawing the house first is commonly linked
to preoccupation with family relationships, security needs,
or domestic concerns. Beginning with the tree is sometimes
interpreted as reflecting interest in vitality, strength, or internal
resources, especially in children [37]. Using the bounding
boxes detected by YOLO and the action timestamps from the
metadata, our module can determine the order and completion
time of each drawn object.
Line Width:The painting interface allows adjustment of
brush and eraser width. Across projective drawing research,
heavy or thick lines are often associated with heightened
tension, aggressive impulses, frustration, or strong emotional
activation. In contrast, very light, faint lines may reflect
anxiety, insecurity, or low energy [38].
In addition to object detection, we use captured metadata to
reconstruct complete drawings. This reconstruction validates
the consistency of the logged data and enables a visual replay
of the drawing process, facilitating quantitative analyzes that
capture the temporal evolution of the sketch rather than only
its final outcome. Figure 5 shows a comparison between an
original drawing and its reconstruction.
(a) Original Drawing
 (b) Reconstructed Drawing
Fig. 5: Original drawing and its reconstruction using metadata.
E. Description Generation
The description generation module acts as a synthesis mod-
ule that converts various data streams into a structured textual
representation of the drawing. It integrates three primary
sources of information including the object detection results,
the classification markers, and the behavioral metadata.
First, the module utilizes the results from the two-stage
object detection to identify which elements are present or omit-
ted. For each detected object, it incorporates the findings from
the six classification models to describe specific qualitative
features, such as a leaning house or a poker face. Second, the
extracted data from drawing metadata is further summarized
into structured textual descriptions.
This structured description serves as the critical bridge
between raw visual/behavioral data and the final psychological
interpretation. By combining drawn object and the drawing
process, the generator provides a comprehensive query for
the subsequent RAG phase. The visualization of drawing
description is illustrated in Figure 6.
Fig. 6: Example of how ArtCognition labels objects and
records drawing behavior.
F . Psychological Interpretation via RAG Architecture
The RAG architecture enables the system to transform raw
drawing descriptions into expert-level psychological insights.
By combining the synthesized data from previous modules
with a curated knowledge base, the system generates grounded
interpretations rather than isolated observations. This approach
ensures that the final output is both evidence-based and aligned
with established clinical frameworks.
1) Data preprocessing:The RAG knowledge base was
constructed from authoritative psychological sources, includ-
ingInterpreting Projective Drawings: A Self-Psychological

6
Approach[35] and the study by Guo et al. [11]. Raw text
was cleaned by removing non-ASCII characters, followed by
stopword removal and lemmatization. To improve retrieval
precision, the corpus was categorized according to section
headings and subheadings, resulting in standardized data op-
timized for downstream tasks.
2) Data chunking:The corpus was segmented using four
strategies to evaluate retrieval performance: a) Character-level
chunking, fixed-length text segments; b) Recursive character
splitting, recursive partitioning along semantic boundaries
(paragraphs, sentences) until a size threshold is reached; c)
Semantic chunking, grouping sentences by embedding similar-
ity; and d) Semantic clustering, applying K-means on sentence
embeddings to form semantically coherent chunks.
3) Generating interpretations with an LLM:We apply
prompt engineering, where retrieved context is provided to
the large language model. Conditioning the LLM on domain-
specific references grounds the generated interpretations in
validated psychological knowledge, mitigates hallucinations,
which is a critical concern in clinical AI, and yields inter-
pretable, context-aware reports.
IV. RESULTS
A. Object Detection
The object detection models were evaluated on a range
of drawing elements, including houses, trees, persons, and
their constituent parts. Table I reports performance across
all classes. The main elements such as house, trees, and
person achieved high precision and recall, whereas smaller
or less frequent elements (e.g., birds) showed lower detection
accuracy.
TABLE I: YOLOv11 performance on main object detection.
Class # P R mAP@50 mAP@50-95
All 103 0.898 0.897 0.949 0.833
Bird 4 1.000 0.710 0.768 0.421
Cloud 15 0.963 0.867 0.956 0.895
Flower 3 0.567 1.000 0.995 0.908
House 21 0.977 1.000 0.995 0.941
Person 22 0.903 0.909 0.968 0.848
Sun 6 0.949 1.000 0.995 0.949
Tree 23 0.886 0.913 0.978 0.902
Chimney Smoke 9 0.942 0.778 0.939 0.802
The detection results for constituent parts are summarized
in Tables II–III-IV. For houses (Table II), windows and roofs
achieved strong performance with mAP@50-95 values of0.92
and0.84, respectively, while doors also showed high accuracy
(0.87). Chimneys exhibited comparatively lower performance
(mAP@50-95= 0.61), likely due to their small size and lim-
ited samples. For people (Table IV), heads were detected most
reliably (mAP@50-95= 0.73), followed by eyes, legs, and
mouths, whereas finer components such as the nose and neck
showed weaker results, reflecting their small spatial extent and
higher variability. For trees (Table III), crowns demonstrated
robust detection performance (mAP@50-95= 0.95), while
trunks and roots achieved moderate accuracy, and branches
and fruit yielded lower scores, consistent with their diverse
shapes and limited annotations.TABLE II: YOLOv11 performance on house part detection.
Class # P R mAP@50 mAP@50-95
All 83 0.965 0.836 0.940 0.808
Chimney 11 0.911 0.636 0.830 0.610
Door 20 0.993 0.850 0.968 0.866
Roof 16 0.987 0.938 0.983 0.837
Window 36 0.971 0.920 0.980 0.917
TABLE III: YOLOv11 performance on tree part detection.
Class # P R mAP@50 mAP@50-95
All 63 0.725 0.931 0.875 0.668
Branches 8 0.495 0.875 0.745 0.494
Crown 23 0.968 1.000 0.995 0.950
Fruit 3 0.316 1.000 0.806 0.672
Root 1 0.928 1.000 0.995 0.597
Trunk 28 0.916 0.782 0.833 0.628
In general, object detection models achieve robust perfor-
mance in main elements and satisfactory detection of con-
stituent parts, providing a reliable foundation for the extraction
of downstream features from HTP drawings.
B. Classification Model
For psychological interpretation, houses, trees, and persons
were annotated across six semantic classification tasks (e.g.,
leaning house vs. non-leaning house). Six models, called
ConvNeXt-Base, ViT-B/16, Swin Transformer, EfficientNet-
B0, ResNet50, and MobileNetV2, were evaluated using ac-
curacy, precision, recall, and F1-score (Tables VII–VIII). The
top-performing model for each task was integrated into the
pipeline.
C. Psychological Interpretation
To ensure the clinical relevance of our framework, we
evaluated the precision of generated description based on
drawing and two core components of the RAG pipeline: the
retrieval of psychological context and the generation of the
final assessment report.
TABLE IV: YOLOv11 performance on person part detection.
Class # P R mAP@50 mAP@50-95
All 182 0.822 0.762 0.825 0.601
Eye 39 0.802 0.949 0.923 0.708
Hand 41 0.794 0.656 0.783 0.649
Head 21 1.000 0.995 0.995 0.731
Leg 39 0.892 0.795 0.884 0.686
Mouth 19 0.813 0.842 0.908 0.746
Neck 14 0.906 0.688 0.827 0.333
Nose 9 0.547 0.406 0.456 0.352
TABLE V: Performance of Leaning House Classifier.
Model Accuracy Precision Recall F1 Score
ConvNeXt-Base 0.952 0.9750.7500.821
ViT-B/16 0.905 0.452 0.500 0.475
Swin Transformer 0.857 0.639 0.697 0.659
EfficientNet-B0 0.810 0.447 0.447 0.447
ResNet50 0.810 0.6670.8950.691
MobileNetV2 0.857 0.450 0.474 0.462

7
TABLE VI: Performance of 2D House Classifier.
Model Accuracy Precision Recall F1 Score
ConvNeXt-Base 0.619 0.683 0.653 0.611
ViT-B/16 0.571 0.607 0.597 0.568
Swin Transformer 0.810 0.846 0.833 0.809
ResNet50 0.476 0.486 0.486 0.476
MobileNetV2 0.714 0.708 0.708 0.708
EfficientNet-B0 0.524 0.543 0.542 0.523
TABLE VII: Performance of Dead Tree Classifier.
Model Accuracy Precision Recall F1 Score
ConvNeXt-Base 0.970 0.978 0.955 0.965
ViT-B/16 0.788 0.879 0.682 0.698
Swin Transformer 0.848 0.907 0.773 0.802
EfficientNet-B0 0.303 0.273 0.250 0.260
ResNet50 0.909 0.907 0.886 0.895
MobileNetV2 0.788 0.879 0.682 0.698
Description Generator Performance:We first evaluate the
accuracy of the description generator, which translates vision-
based detections and drawing dynamics into structured seman-
tic descriptions used as input to the retrieval module. For
each drawing, the generator produces a set of object-level
descriptions corresponding to the core HTP elements (house,
tree, and person). Each description includes spatial localization
(bounding box and placement), categorical attributes (e.g.,
object type, size, and dimensionality), visual properties (e.g.,
presence of windows or doors, color usage), and behavioral
features derived from the drawing process (e.g., drawing order,
stroke speed, and number of actions).
To quantitatively assess description quality, annotators man-
ually reviewed the generated descriptions and counted the
number of incorrectly detected or misattributed features for
each object. A feature was considered incorrect if it was either
falsely detected (e.g., reporting a window when none was
present), omitted despite clear visual evidence, or inaccurately
described (e.g., incorrect placement or size category). Using
these annotations, we compute average precision over the test
dataset, defined as the proportion of correctly detected features
relative to the total number of predicted features.
Across the evaluation set, the description generator achieves
an average precision of 97.57%, indicating a high level of
TABLE VIII: Performance of Flattened Crown Classifier.
Model Accuracy Precision Recall F1 Score
ConvNeXt-Base 0.818 0.474 0.429 0.450
ViT-B/16 0.909 0.476 0.476 0.476
Swin Transformer 0.864 0.475 0.452 0.463
EfficientNet-B0 0.955 0.477 0.500 0.488
ResNet50 0.909 0.476 0.476 0.476
MobileNetV2 0.955 0.477 0.500 0.488
TABLE IX: Performance of Poker Face Classifier.
Model Accuracy Precision Recall F1 Score
ConvNeXt-Base 0.783 0.450 0.429 0.439
ViT-B/16 0.870 0.455 0.476 0.465
Swin Transformer 0.783 0.450 0.429 0.439
ResNet50 0.783 0.450 0.429 0.439
MobileNetV2 0.565 0.433 0.310 0.361
EfficientNet-B0 0.826 0.452 0.452 0.452TABLE X: Performance of Single Line Limb Classifier.
Model Accuracy Precision Recall F1 Score
ConvNeXt-Base 0.8640.7710.819 0.790
ViT-B/16 0.8640.7810.722 0.745
Swin Transformer 0.864 0.771 0.819 0.790
EfficientNet-B0 0.864 0.771 0.819 0.790
ResNet50 0.818 0.675 0.597 0.614
MobileNetV2 0.727 0.542 0.542 0.542
agreement between the generated descriptions and human
annotations. Errors primarily arise in fine-grained visual at-
tributes, such as ambiguous dimensionality (two-dimensional
vs. three-dimensional representations) and subtle structural
elements, while core object identification, spatial placement,
and drawing-order features are detected with consistently high
accuracy. These results suggest that the generated descriptions
provide a reliable and semantically grounded representation
of the drawings, forming a stable foundation for subsequent
retrieval and psychological interpretation.
Retrieval Performance:Second, we assessed the accuracy
of the retriever in fetching relevant psychological interpre-
tations for specific visual descriptions. Ground-truth anno-
tations were established for multiple query sets to measure
the semantic alignment of retrieved chunks. As presented in
Table XI, we compared various chunking strategies. Semantic
chunking yielded the highest cosine similarity (0.991), closely
followed by K-means semantic clustering (0.989). These
methods significantly outperformed character-based splitting,
indicating that semantically coherent segmentation is crucial
for maintaining the integrity of psychological concepts during
the retrieval process.
TABLE XI: RAG cosine similarity by chunking strategy.
Chunking Strategy Cosine Similarity
Character-level chunking 0.978
Recursive character text splitting 0.961
Semantic chunking0.991
Semantic clustering with K-means 0.989
Generative Quality Assurance and Clinical Validity:We
evaluate the quality and reliability of the generated interpreta-
tions through a controlled comparison with two baselines: (1)
a rule-based reporting system following Guo et al. [11], and
(2) a standard gemini-2-flash model operating without retrieval
augmentation. All outputs were anonymized and reviewed in
a blinded setting by licensed clinical psychologists to assess
structural accuracy, theoretical consistency, and practical in-
terpretability.
From a computer vision perspective, the proposed frame-
work demonstrates clear advantages in the structured extrac-
tion of visual information. The model explicitly decomposes
each HTP drawing into semantically meaningful components
(house, tree, and person) and analyzes them using specific
visual attributes, including spatial configuration (e.g., left-
right-center placement), geometric dimensionality (2D vs. 3D
representation), color distribution, and feature completeness.
This structured decomposition addresses the historical critique
of reproducibility in projective testing by replacing impres-
sionistic observation with quantifiable metrics.

8
Crucially, the integration of drawing features with a
retrieval-augmented generation mechanism substantially im-
proves output reliability. While the standard gemini-2-flash
baseline often hallucinates by interpreting features that are not
physically present or assigning meanings not found in the lit-
erature, our framework mitigates this by anchoring its analysis
in validated psychological evidence and the participant’s own
verbal descriptions. To quantitatively assess this, we measured
the hallucination rate, defined as the proportion of interpretive
claims referencing unsupported visual or behavioral features.
The standard gemini-2-flash baseline exhibits a hallucination
rate of approximately 45.72%. By leveraging the RAG archi-
tecture to cross-reference visual detections with established
literature, our framework reduces the error rate to zero, as it
relies strictly on retrieved reference data; any residual errors
originate solely from the object detector or classifier.
Beyond structural accuracy, the generated interpretations
reflect a grounded use of high-level theoretical constructs.
Visual indicators are not merely described but are consistently
linked to psychological states; for instance, spatial orientation
and feature omissions are mapped to constructs such as ego
development, affective constriction, and perceived internal
efficacy.
Furthermore, the model provides clinical value by integrat-
ing pictorial data with the participant’s self-reported ”verbal
attributions” (e.g., the perceived age of the tree or affective
meaning of the house). The system explicitly identifies areas
of convergence and divergence between these modalities. For
example, a discrepancy between a self-reported feeling of
”calmness” and visual markers of depressive tone is high-
lighted not as an error, but as a clinically relevant indicator of
potential defensive processes or limited emotional awareness.
Finally, the interpretations demonstrate meaningful corre-
spondence with standardized psychometric measures. Visual
indicators such as limited color variability, linear limb ge-
ometry, and simplified figures were interpreted in a manner
consistent with elevated scores on the Beck Anxiety Inventory
(BAI) and Beck Depression Inventory (BDI). This align-
ment suggests that the AI-based analysis is not arbitrary, but
possesses convergent validity with established measures of
emotional distress.
Overall, the results indicate that ArtCognition produces
interpretations that are more structured, reproducible, and
evidence-based than ungrounded large language models. While
not a substitute for expert clinical judgment, the framework
enhances transparency by grounding projective analysis in
observable kinematic and visual evidence.
V. DISCUSSION
This study presents a reliable and novel pipeline for auto-
mated HTP analysis that integrates object detection, classifi-
cation, kinematic metadata, and RAG architecture. We discuss
the effectiveness of this approach, limitations of the dataset and
methodology, and ethical considerations of AI-assisted mental
health assessment, outlining directions for future work and the
potential of combining vision models with behavioral metadata
for scalable and interpretable evaluation.A. Clinical Applicability and Trustworthiness
Projective drawing tests are widely used in psychology,
particularly for assessing children who have difficulty verbal-
izing emotions [17], [39]. Prior automated HTP interpretation
methods often relied on non-standardized scoring systems and
handcrafted rules, resulting in inconsistent outputs that can
undermine clinician trust [9], [40]–[43].
ArtCognition addresses these limitations by integrating be-
havioral metadata with visual features to enhance interpretabil-
ity. It generates knowledge-grounded reports through the RAG,
providing clinicians with a transparent and verifiable basis for
analysis while avoiding black-box predictions [44], [45]. By
automating feature extraction and interpretation, the system
reduces the time and effort required for manual scoring [46],
enabling clinicians to focus on treatment planning and inter-
ventions. This efficiency can broaden access to mental health
services, while its adaptability and transparent outputs increase
clinician trust, supporting broader adoption. Moreover, the
design adheres to key principles of trustworthy medical AI,
including explainability, and privacy protection [47].
B. Limitations
Despite promising results, this study faces key limitations,
including a small dataset, missing behavioral modalities, cul-
tural homogeneity, and constraints of automated interpretation.
These challenges point to important directions for future work.
Dataset Size:The dataset is relatively small, limiting model
performance, especially for rare drawing patterns. This scarcity
reduces generalization in both detection and classification
tasks.
Digital Pen Pressure:Pen pressure and tilt were not cap-
tured, omitting insights linked to neuromotor behavior and
limiting the richness of psychological interpretation.
Cultural and Demographic Homogeneity:All participants
were selected from a single cultural background, which may
introduce bias and limit the generalizability of the models to
broader populations.
Interpretive Limitations:Although high-resolution features
of drawings can be quantified with precision, there is limited
clinical evidence to support reliable interpretation of many fea-
tures. Moreover, certain psychological nuances remain difficult
to capture computationally, highlighting the continued need for
expert supervision.
C. Ethical Considerations
Automated psychological assessment raises important con-
cerns regarding privacy and responsible use. Addressing these
issues is critical for the safe and ethical deployment of digital
mental health applications.
VI. CONCLUSION
We present a state-of-the-art AI pipeline for automated
analysis of digital HTP drawings, integrating object detection,
classification, behavioral metadata, and a RAG architecture
for psychological interpretation. By combining visual features,
kinematic data, and external psychological knowledge, the

9
system provides interpretable and scalable assessments that
go beyond traditional manual scoring. Experimental results
demonstrate high detection and classification accuracy, in addi-
tion metadata analysis captures nuanced behavioral indicators
such as drawing speed, line smoothness, and drawing order.
Key limitations include a small, culturally homogeneous
dataset, the absence of pen pressure data, and the inherent
challenge of quantifying complex psychological states. Future
work will expand the dataset, incorporate additional neuromo-
tor modalities, and validate the approach across diverse pop-
ulations. Overall, this study demonstrates the potential of AI-
assisted digital drawing analysis as a complementary tool for
psychological assessment, offering efficiency, interpretability,
and evidence-based insights.
ACKNOWLEDGMENTS
The authors would like to thank Shayan Talebiani for
his contributions to data labeling and for implementing the
metadata-based module used to compute the number of actions
associated with each object.
APPENDIXA
DATASET
This appendix presents comprehensive dataset statistics,
detailing the frequency of object classes and their constituent
parts. These metrics are provided to facilitate reproducibility
and offer essential context for the experimental results and
analyses discussed in the main text. Table XII presents the
frequency of the main object categories in the dataset, showing
how often each high-level object (e.g., house, tree, person)
appears. It provides an overview of the dataset’s overall
composition and helps illustrate class balance.
TABLE XII: Distribution of main object classes in the dataset.
Class # Instances
Bird 66
Cloud 112
House 147
Tree 181
Person 156
Flower 46
Mountain 14
Sun 63
Chimney Smoke 45
TABLE XIII: Distribution of house, tree, and person object
classes in the dataset.
(a) House
Class # Inst.
Door 135
Window 249
Roof 107
Chimney 58(b) Tree
Class # Inst.
Branches 40
Crown 158
Fruit 144
Root 21
Trunk 180(c) Person
Class # Inst.
Eye 213
Hand 266
Head 134
Leg 258
Mouth 105
Neck 74
Nose 37
APPENDIXB
HOUSE–TREE–PERSONQUESTIONNAIRE
This questionnaire provides additional contextual informa-
tion about participants’ interpretations of their drawings. It wasdesigned by a clinical expert to capture subjective perceptions
related to the house, tree, and person figures.
A. Participant Information
Name: Age: Gender:
B. Questions
1) Who do you imagine lives in this house?
2) What feelings does this house give you?
3) How old do you think the tree is?
4) Is the tree alive or dead?
5) Which season of the year do you think it is?
6) Does this image remind you of anyone?
7) How old do you think this person is?
8) What do you think this person does?
9) What might this person be thinking?
10) What do you think this person feels?
APPENDIXC
SAMPLE OF DRAWING INTERPRETATION USINGLLM
Figure 7 shows the drawing used as the primary input
for sample analysis. While the vanilla model only looks at
the static image, ArtCognition integrates various data layers
to generate a more comprehensive evaluation. The following
sections compare the basic interpretation with the advanced
output provided by ArtCognition.
Fig. 7: Sample drawing used to compare vanilla and ArtCog-
nition interpretation methods.
A. Vanilla Interpretation
1) Overall Impression:The drawing presents a simple,
colorful scene of a house, tree, and person, suggesting a
focus on basic elements of life and environment. The bright
colors and smiling face indicate a generally positive affect.
The composition appears somewhat spread out across the page,
potentially suggesting a need for more structure or boundaries.
2) House:
•Color: The pink color of the house could represent
warmth, affection, or a feminine association.
•Shape: The house is drawn in a simple, conventional
form, suggesting adherence to norms and a desire for
stability.
•Windows and Door: The presence of windows and a door
indicates openness to social interaction and a desire to

10
connect with the outside world. The windows are large
and clearly defined, further emphasizing this openness.
•Foundation: The house appears to be grounded by a wavy
green line, possibly representing the environment and a
sense of connection to nature.
3) Tree:
•Size and Shape: The tree is large and has a full, rounded
crown, suggesting a strong sense of growth, vitality, and
emotional expression.
•Trunk: The trunk is depicted with two lines, which could
indicate strength and stability.
•Placement: The tree is positioned between the house and
the person, potentially symbolizing a bridge between the
home environment and the individual’s self-expression.
4) Person:
•Facial Expression: The smiling face is a clear indicator
of happiness and a positive outlook.
•Hair: The curly hair may represent energy, creativity, or
a desire for attention.
•Arms and Legs: The arms are extended, suggesting a
willingness to engage with the world. The legs are simple,
indicating a sense of groundedness.
•Clothing: The dress could represent femininity or a sense
of self-presentation.
•Shoes: The pink shoes add a touch of whimsy and
individuality.
5) Color Usage:The use of bright colors like pink, green,
and red suggests optimism, energy, and a desire for stimula-
tion.
6) Placement on the Page:The drawing is positioned in
the lower half of the page, which could indicate a sense of
security or a need for support.
7) Limitations:It’s crucial to remember that this is a basic
interpretation based solely on the drawing. A comprehen-
sive analysis would require additional information about the
artist’s background, experiences, and emotional state. This
interpretation should be considered a starting point for further
exploration, not a definitive diagnosis.
B. ArtCognition’s Interpretation
1) Overall Impression:The drawing, featuring a house,
tree, and person, is centrally placed, suggesting a feeling
of security and a capacity to relate to the present moment.
However, the exact center placement can also evoke a need
for balance and rigidity. The simplified nature of the drawing,
coupled with the lack of additional decoration, may indicate
low mental motivation or a tendency towards avoidance and
retreat. The use of normal coloring suggests the drawer can
express a wide range of subjective experience.
2) House:
•General: The three-dimensional house signifies an at-
tempt to represent depth. The drawer identifies with
the house as her childhood home, associating it with
happiness and a lack of sadness, indicating a potentially
positive view of her past.•Door: The presence of a door suggests a degree of acces-
sibility, but without a doorknob, it conveys uncertainty or
ambivalence about letting others gain access.
•Windows: The presence of windows indicates a capacity
for relatedness and interaction with the surroundings.
•Color: The pink color of the house is not directly
referenced in the provided text, thus hard to provide
interpretation.
3) Tree:
•General: As a tree is alive, and depicts summer, it
suggests a sense of vitality and aliveness. The drawer
sees the tree as 20 years old, while she is 29, and there
is a diminished sense of felt maturity.
•Color: The green color evokes aliveness and expansive-
ness.
4) Person:
•General: The complete person with appropriate body
proportions suggests an intact sense of self-cohesion. The
moderate size of the person indicates an optimal sense of
adequacy. The figure reminds the drawer of herself as a
little girl, reinforcing the connection to childhood.
•Facial Features: The presence of facial features indicates
a concern about appearance and how one presents oneself
to others. However, the omission of the nose suggests
a serious disturbance in how one appears to others and
receives affective stimuli.
•Limbs: The inclusion of limbs (legs and hands) indicates
a capacity for maneuverability and connection with the
environment.
•Color: The use of pink on the person again has no direct
meaning, but a general observation can be made: chro-
matic drawings allows for a wider range of expression.
5) Color:
•Green: The use of green suggests a sense of aliveness
and expansiveness.
•White: The white background alludes a loss of self-
awareness and psychological defenses.
•Pink: Color is not referenced in the provided text.
6) Placement:The central page placement indicates nor-
mality.
7) Summary:The drawing suggests a connection to the
drawer’s childhood, associating it with happiness. There is
a sense of vitality and a desire for connection, but also
some ambivalence about accessibility and a tendency toward
withdrawal, possibly as a defense. The omission of the nose
and the simplified style suggest some underlying anxieties
related to self-perception and interaction with the world. The
central placement and moderate size of the figures indicate a
basic sense of security and adequacy, although the need for
balance and control may be a factor.
C. Accuracy Score Computation
To quantify interpretive hallucinations, we measure an ac-
curacy score based on expert validation. For each generated
interpretation, individual claims are decomposed into atomic
interpretation units. Each unit is labeled ascorrectorin-
correctaccording to domain-expert (psychologist) feedback

11
and established HTP methodological rules. Omissions are
not penalized unless replaced by an incorrect inference. For
example, in a representative case, 13 interpretation units were
identified, of which 5 were judged correct and 8 were judged
incorrect due to unsupported symbolic or affective inferences.
The final accuracy score is approximately38.5%, where lower
value indicate a higher propensity for hallucination in vanilla
interpretation.
APPENDIXD
HOUSE–TREE–PERSONDRAWINGCHARACTERISTICS
Table XIV summarizes detailed information on the drawing
characteristics analyzed in the House–Tree–Person test and
their associated psychological interpretations. We use this
table to compare our method with the study by Guo et
al. [11], highlighting the observable drawing features and the
corresponding inferred psychological constructs.
TABLE XIV: Drawing characteristics and their associated
psychological interpretations.
Drawing Characteristics Indicates Meaning
Excessive separation among items; Omitted
house, tree, or person; No door / window;
Loss of facial features / poker face; Complete
or partial loss of limbs; Incomplete person;
Left page placement / upper-left corner place-
ment; Color: whiteLoss of self-awareness and psycho-
logical defenses
Leaning house; dead Tree; flattened crown;
Inappropriate body proportions; Fist; High or
right page placement; Colors: purple, brownPsychological conflict and sense of
unreality
Smoking chimney; Roots; Colors: yellow,
purple; Top edge page placementNervousness, sensitivity, and irri-
tability
No additional decoration; Simplified draw-
ing; Small drawing size; Emphasis on straight
lines; Very small house / tree / person; Two-
dimensional house; Single line limbs; Ab-
sence of color; Low page placement; Color:
white; Faint lines; Left page placementLow mental motivation, avoidance,
and retreat
Left page placement; Low page placement;
Colors: brown, white; Upper-left corner
placementRegression
Central page placement; Colors: orange,
green, blueNormality
Low page placement; Small drawing size;
Very faint lines; Color: whiteDepression, emptiness
Low page placement; Small drawing size;
Color: brown; Left/top edge page placementInsecurity
Large drawing size; Heavy/thick lines Aggression
Left page placement; Small drawing size;
Very faint lines; Low page placement; Color:
greenSelf-esteem, childish
Bottom edge of paper Need for external support, depen-
dence
Side edge of paper; Large drawing size; Ex-
cessive use of colorEnvironmental restriction, pressure
Large drawing size; Colors: red, orange; High
page placementHeightened vitality, energy, manic
states
REFERENCES
[1] W. H. Organization, “Mental disorders fact sheets,” https://www.who.
int/news-room/fact-sheets/detail/mental-disorders, 2019, accessed: Sep.
11, 2025.[2] A. M. Kilbourne, K. Beck, B. Spaeth-Rublee, P. Ramanuj, R. W.
O’Brien, N. Tomoyasu, and H. A. Pincus, “Measuring and improving the
quality of mental health care: a global perspective,”World psychiatry,
vol. 17, no. 1, pp. 30–38, 2018.
[3] F. Hanna, C. Barbui, T. Dua, A. Lora, M. van Regteren Altena, and
S. Saxena, “Global mental health: how are we doing?”World Psychiatry,
vol. 17, no. 3, p. 367, 2018.
[4] L. Balcombe and D. De Leo, “Digital mental health challenges and
the horizon ahead for solutions,”JMIR Mental Health, vol. 8, no. 3, p.
e26811, 2021.
[5] G. C. Gleser, “Projective methodologies,”Annual review of psychology,
vol. 14, no. 1, pp. 391–422, 1963.
[6] J. N. Buck, “The htp technique; a qualitative and quantitative scoring
manual.”Journal of clinical psychology, 1948.
[7] W. J. Camara, J. S. Nathan, and A. E. Puente, “Psychological test us-
age: Implications in professional psychology.”Professional psychology:
Research and practice, vol. 31, no. 2, p. 141, 2000.
[8] H. Smeijsters and G. Cleven, “The treatment of aggression using arts
therapies in forensic psychiatry: Results of a qualitative inquiry,”The
arts in psychotherapy, vol. 33, no. 1, pp. 37–58, 2006.
[9] L. Sheng, G. Yang, Q. Pan, C. Xia, and L. Zhao, “Synthetic house-
tree-person drawing test: A new method for screening anxiety in cancer
patients,”Journal of Oncology, vol. 2019, no. 1, p. 5062394, 2019.
[10] T. Ostermann, J. P. R ¨oer, and M. J. Tomasik, “Digitalization in psychol-
ogy: A bit of challenge and a byte of success,”Patterns, vol. 2, no. 10,
2021.
[11] H. Guo, B. Feng, Y . Ma, X. Zhang, H. Fan, Z. Dong, T. Chen, and
Q. Gong, “Analysis of the screening and predicting characteristics of
the house-tree-person drawing test for mental disorders: A systematic
review and meta-analysis,”Frontiers in psychiatry, vol. 13, p. 1041770,
2023.
[12] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look
once: Unified, real-time object detection,” inProceedings of the IEEE
conference on computer vision and pattern recognition, 2016, pp. 779–
788.
[13] M. Tan and Q. Le, “Efficientnet: Rethinking model scaling for con-
volutional neural networks,” inInternational conference on machine
learning. PMLR, 2019, pp. 6105–6114.
[14] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gellyet al.,
“An image is worth 16x16 words: Transformers for image recognition
at scale,”arXiv preprint arXiv:2010.11929, 2020.
[15] Z. Liu, Y . Lin, Y . Cao, H. Hu, Y . Wei, Z. Zhang, S. Lin, and
B. Guo, “Swin transformer: Hierarchical vision transformer using shifted
windows,” inProceedings of the IEEE/CVF international conference on
computer vision, 2021, pp. 10 012–10 022.
[16] J. Feng, H. Tan, W. Li, and M. Xie, “Conv2next: reconsidering conv next
network design for image recognition,” in2022 international conference
on computers and artificial intelligence technologies (CAIT). IEEE,
2022, pp. 53–60.
[17] M. Lee, Y . Kim, and Y .-K. Kim, “Generating psychological analysis
tables for children’s drawings using deep learning,”Data & Knowledge
Engineering, vol. 149, p. 102266, 2024.
[18] A. Alshahrani, M. M. Almatrafi, J. I. Mustafa, L. S. Albaqami, and
R. A. Aljabri, “A children’s psychological and mental health detection
model by drawing analysis based on computer vision and deep learning,”
Engineering, Technology & Applied Science Research, vol. 14, no. 4, pp.
15 533–15 540, 2024.
[19] S. Dhingra, M. Singh, N. Malviya, S. S. Gillet al., “Mind meets ma-
chine: Unravelling gpt-4’s cognitive psychology,”BenchCouncil Trans-
actions on Benchmarks, Standards and Evaluations, vol. 3, no. 3, p.
100139, 2023.
[20] “Htp painter (v1.2),” https://github.com/behradbina/Paint, 2025, [Online;
accessed 19-Aug-2025].
[21] M. Takahashi, “Introduction to the drawing test: The htp test,”Korea
Instute of Color Psychological Analysis, pp. 29–34, 2009.
[22] M. Sandler, A. G. Howard, M. Zhu, A. Zhmoginov, and L.-C.
Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,”2018
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 4510–4520, 2018. [Online]. Available: https://api.semanticscholar.
org/CorpusID:4555207
[23] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for
image recognition,”2016 IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), pp. 770–778, 2015. [Online]. Available:
https://api.semanticscholar.org/CorpusID:206594692
[24] M. C. Meehan, “Psychological evaluation of children’s human figure
drawings,”JAMA, vol. 205, no. 3, pp. 190–190, 1968.

12
[25] S. Rueckriegel, F. Blankenburg, R. Burghardt, S. Ehrlich, G. Henze,
R. Mergl, and P. Driever, “Influence of age and movement complexity
on kinematic hand movement parameters in childhood and adolescence,”
International journal of developmental neuroscience : the official journal
of the International Society for Developmental Neuroscience, vol. 26, pp.
655–63, 09 2008.
[26] A. Abu-Ata, D. Green, R. Sopher, S. Portnoy, and N. Ratzon, “Upper
limb kinematics of handwriting among children with and without
developmental coordination disorder,”Sensors, vol. 22, p. 9224, 11 2022.
[27] Y . Yamada, M. Kobayashi, M. Nemoto, M. Ota, K. Nemoto, and T. Arai,
“Characteristics of drawing process differentiate alzheimer’s disease and
dementia with lewy bodies,”Journal of Alzheimer’s Disease, vol. 90, pp.
693–704, 09 2022.
[28] A. Scholp, M. Hoffman, S. Rosen, M. Abdelhalim, C. Jones, J. Jiang,
and T. Mcculloch, “Spectral arc length as a method to quantify pha-
ryngeal high-resolution manometric curve smoothness,”Neurogastroen-
terology & Motility, vol. 33, 04 2021.
[29] A. Barrett and C. Craver-Lemley, “Is it what you see, or how you say
it? spatial bias in young and aged subjects,”Journal of the International
Neuropsychological Society : JINS, vol. 14, pp. 562–70, 08 2008.
[30] D. Picard and B. Zarhbouch, “Leftward spatial bias in children’s draw-
ing placement: Hemispheric activation versus directional hypotheses,”
Laterality: Asymmetries of Body, Brain and Cognition, vol. 19, no. 1,
pp. 96–112, 2014.
[31] E. Apostolopoulou and A. Klonari, “Pupils’ representations of rivers on
2d and 3d maps,”Procedia-Social and Behavioral Sciences, vol. 19, pp.
443–449, 2011.
[32] J. N. Buck,The house-tree-person technique: Revised manual. Western
Psychological Services, 1966.
[33] J. Buck and S. Sloan,The House-Tree-Person Technique Revised
Manual. Ishi Press International, 2019. [Online]. Available: https:
//books.google.com/books?id=zlZ0yAEACAAJ
[34] J. Christie, M. Reichertz, B. Maycock, and R. M. Klein, “To erase or
not to erase, that is not the question: Drawing from observation in an
analogue or digital environment,”art, design & communication in higher
Education, vol. 19, no. 2, pp. 203–220, 2020.
[35] M. Leibowitz,Interpreting projective drawings: A self-psychological
approach. Routledge, 2013.
[36] G. Groth-marnat, “The handbook of psychological assessment,” vol. 3rd,
01 2009.
[37] L. Handler and A. Thomas,Drawings in Assessment and Psychotherapy:
Research and Application. Taylor & Francis, 2013. [Online]. Available:
https://books.google.com.br/books?id=OtwkAgAAQBAJ
[38] M. Cooper,Line Drawing Interpretation, ser. Economic studies in in-
equality, social exclusion and well-being. Springer London, 2010. [On-
line]. Available: https://books.google.nl/books?id=pmAQxCUg12UC
[39] N. Ali, A. Abd-Alrazaq, Z. Shah, M. Alajlani, T. Alam, and M. Househ,
“Artificial intelligence-based mobile application for sensing children
emotion through drawings,” inAdvances in Informatics, Management
and Technology in Healthcare. IOS Press, 2022, pp. 118–121.
[40] W. CAI, Y .-L. TANG, S. WU, and Z.-Z. CHEN, “The tree in the
projective tests,”Advances in Psychological Science, vol. 20, no. 5, p.
782, 2012.
[41] G. Chen and W. Yan, “Utility of the rorschach inkblot test in clinical
psychological diagnosis,”China J Health Psychol, vol. 30, pp. 475–80,
2022.
[42] H. Zhou, “Research on the relationship between rumination thinking of
junior middle school students and htp drawing characteristics,”Jinzhou:
Bohai University, 2021.
[43] J. Xiang, M. Liao, and M. Zhu, “Assessment of junior elementary pupils’
depression tendency via house-tree-person test,”China J Health Psychol,
vol. 28, pp. 1057–61, 2020.
[44] D. W. Joyce, A. Kormilitzin, K. A. Smith, and A. Cipriani, “Explainable
artificial intelligence for mental health through transparency and inter-
pretability for understandability,”npj Digital Medicine, vol. 6, no. 1,
p. 6, 2023.
[45] S. Tutun, M. E. Johnson, A. Ahmed, A. Albizri, S. Irgil, I. Yesilkaya,
E. N. Ucar, T. Sengun, and A. Harfouche, “An ai-based decision support
system for predicting mental health disorders,”Information Systems
Frontiers, vol. 25, no. 3, pp. 1261–1276, 2023.
[46] M. O. Zeeshan, I. Siddiqi, and M. Moetesum, “Two-step fine-tuned
convolutional neural networks for multi-label classification of children’s
drawings,” inInternational Conference on Document Analysis and
Recognition. Springer, 2021, pp. 321–334.
[47] M. Kim, H. Sohn, S. Choi, and S. Kim, “Requirements for trustworthy
artificial intelligence and its application in healthcare,”Healthcare
Informatics Research, vol. 29, pp. 315–322, 10 2023.