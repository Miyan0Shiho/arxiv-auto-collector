# TimeLens: On-Device Artifact Recognition with Retrieval-Augmented Question Answering for the Grand Egyptian Museum

**Authors**: Rawan Hesham, Ali Ashraf, Amr Ahmed, Malak Alaa, Omar Ahmed, Omar Wagih

**Published**: 2026-06-11 12:23:46

**PDF URL**: [https://arxiv.org/pdf/2606.13267v1](https://arxiv.org/pdf/2606.13267v1)

## Abstract
TimeLens is an AI-powered bilingual mobile guide for the Grand Egyptian Museum (GEM). Pointing a phone at an exhibit, a visitor sees the artifact recognized in real time and can ask follow-up questions answered in English or Arabic. The work addresses three problems specific to in-gallery deployment: fine-grained visual similarity among 51 catalogued artifacts (many near-identical Ramesside statues), the gap between curated training data and handheld camera conditions, and the risk of an AI guide stating unsupported historical facts. Two engineering contributions are reported. First, an on-device artifact detector was developed through a data-quality-driven iteration study -- from foundation-model auto-annotation (YOLO-World), through spatial label-cleaning rules, to a fully hand-annotated dataset -- isolating label quality as the decisive factor: the final YOLOv8n model resolves every previously failing class while remaining a 5.97 MB TensorFlow Lite asset that runs in real time on a mid-range phone (mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.924). Second, a bilingual Retrieval-Augmented Generation (RAG) guide, grounded in a 108-record ChromaDB knowledge base, was benchmarked across seven candidate language models, with Gemma 4 E2B (Q4 K M) selected; ten targeted optimizations reduce end-to-end latency from over 30 s to approximately 10 s. Both subsystems are integrated in a production Flutter application with bilingual interface, museum location gating, and text-to-speech support.

## Full Text


<!-- PDF content starts -->

TimeLens: On-Device Artifact Recognition with
Retrieval-Augmented Question Answering
for the Grand Egyptian Museum
1stRawan Hesham
Faculty of Computers and AI
Capital University
Cairo, Egypt
rawan 20220167@fci.capu.edu.eg2ndAli Ashraf
Faculty of Computers and AI
Capital University
Cairo, Egypt
AliEldin 20220296@fci.capu.edu.eg3rdAmr Ahmed
Faculty of Computers and AI
Capital University
Cairo, Egypt
Amr 20210632@fci.capu.edu.eg
4thMalak Alaa
Faculty of Computers and AI
Capital University
Cairo, Egypt
Malak 20220503@fci.capu.edu.eg5thOmar Ahmed
Faculty of Computers and AI
Capital University
Cairo, Egypt
Omar 20220310@fci.capu.edu.eg6thOmar Wagih
Faculty of Computers and AI
Capital University
Cairo, Egypt
Omar 20220324@fci.capu.edu.eg
Abstract—TimeLens is an AI-powered bilingual mobile guide
for the Grand Egyptian Museum (GEM). Pointing a phone at
an exhibit, a visitor sees the artifact recognized in real time and
can ask follow-up questions answered in English or Arabic. The
work addresses three problems specific to in-gallery deployment:
fine-grained visual similarity among 51 catalogued artifacts
(many near-identical Ramesside statues), the gap between curated
training data and handheld camera conditions, and the risk of an
AI guide stating unsupported historical facts. Two engineering
contributions are reported. First, an on-device artifact detector
was developed through a data-quality-driven iteration study —
from foundation-model auto-annotation (YOLO-World), through
spatial label-cleaning rules, to a fully hand-annotated dataset —
isolating label quality as the decisive factor: the final YOLOv8n
model resolves every previously failing class while remaining
a 5.97 MB TensorFlow Lite asset that runs in real time on a
mid-range phone (mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.924).
Second, a bilingual Retrieval-Augmented Generation (RAG)
guide, grounded in a 108-record ChromaDB knowledge base,
was benchmarked across seven candidate language models, with
Gemma 4 E2B (Q4 KM) selected; ten targeted optimizations
reduce end-to-end latency from over 30 s to approximately
10 s. Both subsystems are integrated in a production Flutter
application with bilingual interface, museum location gating, and
text-to-speech support.
Index Terms—on-device object detection; YOLOv8n; Tensor-
Flow Lite; retrieval-augmented generation; Gemma; ChromaDB;
bilingual question answering; augmented reality; Flutter; cul-
tural heritage computing; Grand Egyptian Museum
I. INTRODUCTION
Fine-grained artifact recognition in museum environments
is difficult due to background clutter, low-light capture, visual
overlap between classes, and open-set uncertainty. At the
Grand Egyptian Museum (GEM) the challenge is acute: the
target collection contains 51 catalogued artifacts, many of
which are near-identical Ramesside statues differing only insubtle pose, headdress, or accompanying standards. A tourist
standing in front of two visually similar statues does not care
about model internals; they care whether the app gives a
trustworthy answer quickly and in their own language.
Current museum-technology offerings fail on at least one
of three requirements for GEM deployment: real-time on-
device recognition (cloud APIs introduce latency and con-
nectivity dependence), Arabic-language support (most systems
target Western audiences), and hallucination-free answers (un-
grounded language models routinely invent plausible-sounding
historical facts) [1]. The global Augmented Reality market
exceeded USD 50 billion in 2023 and is projected to surpass
USD 110 billion by 2027 [14]; Egypt welcomed approximately
14.9 million international visitors in 2023 [15], and over 70%
of tourists rely on digital tools during visits [16], confirming
strong demand for a solution that satisfies all three require-
ments simultaneously.
TimeLensaddresses this gap through two engineering con-
tributions. First, a compact YOLOv8n detector [5] is trained on
a carefully curated, video-derived dataset of GEM artifacts and
exported to TensorFlow Lite with end-to-end non-maximum
suppression, enabling real-time on-device recognition with no
per-frame network round-trip. Second, a bilingual Retrieval-
Augmented Generation (RAG) guide [7] grounds every answer
in a curated 108-record ChromaDB knowledge base, prevent-
ing fabrication and supporting both English and Arabic. Both
subsystems are integrated in a production Flutter application.
The key findings of this paper are: (i) label quality domi-
nates model architecture and input resolution for fine-grained
museum artifact detection; (ii) a fully local bilingual RAG
pipeline achieves hallucination-free answers at interactive la-
tency; and (iii) the combined system runs on commodity
phones and a single small GPU, making it suitable for realarXiv:2606.13267v1  [cs.CV]  11 Jun 2026

museum deployment.
II. RELATEDWORK
A. On-Device Object Detection for Cultural Heritage
Single-stage detectors of the YOLO family provide a strong
speed/accuracy balance for mobile deployment and export
cleanly to TensorFlow Lite for on-device inference [5]. Com-
parative studies of museum-exhibit recognition from video-
derived datasets report that YOLOv8 reaches very high mean
Average Precision while remaining light enough for real-
time mobile use, outperforming heavier backbones such as
VGG-16 and ResNet on the speed/accuracy trade-off [2].
Adaptive CNN approaches have likewise been used to enrich
museum interaction through automatic artifact recognition [3].
EfficientNet-based systems have been applied to cultural-
landmark recognition in smart tourism with strong results [4].
B. Foundation-Model Auto-Annotation
Manually annotating tens of thousands of video frames
is prohibitive for a student-led project, motivating auto-
annotation with open-vocabulary, text-prompted detectors such
as YOLO-World [6]. Such foundation models handle generic
concepts well but degrade on fine-grained, domain-specific
objects — precisely the near-identical statues that dominate
the GEM collection — producing systematically mislocalized
or sibling-confused labels. This limitation is the central moti-
vation for our data-quality study.
C. Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) grounds a lan-
guage model’s output in passages retrieved from a knowledge
base rather than its parametric memory, reducing fabrication
in knowledge-intensive tasks [7]. Recent work brings context-
aware LLM assistants into AR settings [8]. Quantized models
run locally through runtimes such as Ollama [9], while multi-
lingual sentence embeddings make a single bilingual retrieval
index practical. Robustness challenges specific to heritage
capture — low contrast, reduced lighting, and crowd occlusion
— are active research topics [10], [11].
D. Research Gaps
Prior work leaves three combined gaps that TimeLens
addresses: (i) label quality for fine-grained, domain-specific
detection; (ii) honest evaluation of video-frame datasets where
random splits leak near-duplicate frames; and (iii) grounded,
bilingual generation for public-facing museum guides.
III. SYSTEMARCHITECTURE
TimeLens separates concerns across two principal layers
with a clear boundary. On the phone, anon-device perception
layerturns each camera frame into a tracked, classified artifact.
Aknowledge layerthen grounds any follow-up question in
curated museum data before the language model phrases the
answer.Mobile App
(Flutter)
YOLOv8n TFLiteFastAPI
RAG ServiceOllama
Gemma 4 E2B
ChromaDB
EN/AR
collectionsHTTPS
answerprompt
retrieve
Fig. 1. System block diagram. Detection is fully on-device; only chat
questions reach the FastAPI RAG service.
A. High-Level Block Diagram
Fig. 1 shows the runtime data flow. Detection runs entirely
on the phone; only a chat question crosses the network,
reaching the FastAPI service that retrieves from ChromaDB
and prompts the locally served Gemma model.
B. Scan-to-Answer Sequence
Fig. 2 traces one interaction from end to end. Identity is
decided on-device the moment the visitor frames an artifact;
the backend is contacted only when a question is asked.
Visitor Mobile App FastAPI ChromaDB+Gemma
point/ask
on-device detect+classify
POST /ask
retrieve
context
answer
display+TTS
Fig. 2. Scan-to-answer sequence. Identity is established on-device; the
backend retrieves context and phrases the grounded answer.
C. On-Device Inference Pipeline
Each camera frame arrives as sensor-rotated YUV420 data,
converted to RGB, rotated90◦to match the Android sensor
orientation, resized to640×640, and normalized before infer-
ence. The YOLOv8n model is exported with NMS baked in, so
a single forward pass returns a fixed-size[1,300,6]tensor
of(xyxy, score, class)rows — removing any post-
processing decode on the device. A confidence filter (global
threshold 0.75, raised to 0.90 for the visually near-identical
Statue of Siptah) then keeps only trustworthy detections.
D. Technology Stack
•Mobile app:Flutter with Provider state management.
•On-device detection:YOLOv8n (Ultralytics) exported to
TensorFlow Lite (FP16) with end-to-end NMS.

•RAG backend:FastAPI + LlamaIndex + LangChain;
Ollama serving Gemma 4 E2B (Q4 KM); ChromaDB
vector store; multilingual MiniLM embeddings.
•Training:Ultralytics 8.4.45 / PyTorch; YOLO-World for
auto-annotation; Roboflow for hand annotation; NVIDIA
GTX 1650 Ti (4 GB).
•Serving:RunPod NVIDIA RTX A5000 (24 GB VRAM)
via ngrok tunnel.
IV. ON-DEVICEARTIFACTDETECTION
A. Dataset and Constraints
The source material consists of approximately 50 in-gallery
videos of GEM artifacts captured before public opening,
yielding roughly 30,000 candidate frames across 51 classes
at about one frame per second. Deployment constraints rule
out cloud detectors: inference must run on-device with no per-
frame round-trip, in real time (≥2.5FPS on mid-tier phones),
with a compact model (<8MB TFLite asset), and be robust
to handheld capture. These constraints led to YOLOv8-nano
with FP16 TensorFlow Lite deployment [2], [5].
B. Label-Quality Iteration Study
Manual annotation of 30k images is infeasible for a student
team, motivating an auto-annotation-first pipeline followed by
progressive quality improvement.
v1 — Auto-Annotated Baseline.Labels were produced
by YOLO-World (yolov8s-worldv2) [6] using per-class
text prompts. A YOLOv8n model trained on these labels at
320×320reached mAP@0.5= 0.751, but per-class analysis re-
vealed ten classes below mAP@0.5= 0.30. Two failure modes
appeared: Mode A (nine classes), where YOLO-World boxed
the wrong object (a wall card, a bench, a neighboring statue);
and Mode B (one class), sibling-class confusion between a
bust and full-body statues. The architecture was capable; the
labels were the bottleneck.
v2 — Spatial Sanity Rules.Four geometric rules were
applied to the YOLO-World output: (1) restrict the “stela”
prompt to the two true stela classes; (2) drop boxes whose
center lies in the bottom quarter of the frame (c y>0.75),
which are almost always floor signs or bystanders; (3) pick
the largest-area box rather than the highest-confidence one; (4)
drop boxes whose center deviates from the per-video median
by more than 0.20. These rules dropped 6.9% of frames
(leaving 27,134), raised mAP@0.5 to 0.867 (+0.116 from a
labeling change alone), and reduced broken classes from ten
to three.
v2@640 — Resolution and End-to-End NMS.Raising
input to640×640and exporting with end-to-end NMS left
mAP@0.5 unchanged at 0.867 but lifted mAP@0.5:0.95 from
0.586 to 0.777 (+0.191): higher resolution improves localiza-
tion but cannot fix classes whose labels are wrong. The data-
quality ceiling was unmistakable.
v3 — Full Hand Annotation.All 51 classes were re-
annotated by hand in Roboflow (24,384 images, 70/20/10
split), drawing full-artifact boxes, avoiding peripheral objects,
and carefully distinguishing similar artifacts. Warm-startedfrom v2@640 weights and trained for 100 epochs on a GTX
1650 Ti, v3 reached precision 0.998, recall 1.000, mAP@0.5
0.995, and mAP@0.5:0.95 0.924, with zero broken classes.
The validation classification loss collapsed from 0.892 to
0.166.
C. Auto-Annotation Algorithm
Algorithm 1 formalizes the v1→v2 spatial-cleaning step.
Algorithm 1:Auto-annotation with spatial sanity rules
(v1→v2)
Input:video framesF; per-class text promptsP; true-stela
class setS
Output:one labeled box per usable frame
foreachframef∈F(classcfrom its folder)do
B←YOLOWORLD(f, P);
ifc /∈Sthen
remove stela-prompt detections fromB
end
remove boxes with centerc y>0.75;
ifB̸=∅then
b⋆←arg max b∈Barea(b);
dropb⋆if its center deviates>0.20from per-video
median;
ifb⋆keptthen
emit(b⋆, c)
end
end
end
D. Detection Results
Table I summarizes the four iterations. Fig. 3 plots the
mAP@0.5 progression. The dominant lesson is that label
cleaning (+0.116 mAP@0.5, no model change) and hand
annotation (+0.128 mAP@0.5, +0.147 mAP@0.5:0.95) broke
ceilings that resolution scaling could not.
TABLE I
DETECTORITERATIONCOMPARISON(VALIDATIONSET)
Iter. Labels imgsz mAP 50mAP 50:95
v1 Auto (YOLO-World) 320 0.751 —
v2 Auto + 4 rules 320 0.867 0.586
v2@640 Auto + rules 640 0.867 0.777
v3 Hand-annotated 640 0.995 0.924
Table II shows six representative classes rescued by hand
annotation; zero classes remain broken in v3.
E. Deployment Performance
The deployed asset is FP16 TFLite at5.97 MB, running
in approximately 800 ms per frame on a mid-tier Android
phone (about 1.25 FPS). Although this falls below the initial
2.5 FPS target, inference was executed asynchronously on a
separate thread, keeping the camera preview and UI responsive
and non-blocking. During field testing no misclassifications
were observed across all 51 artifact classes, with detection
confidence consistently ranging between 75% and 95%. Clean

v1 v2@320 v2@640 v300.51
0.750.87 0.871mAP@0.5
Fig. 3. Detection mAP@0.5 across iterations. Label cleaning (v1→v2) and
hand annotation (v2@640→v3) drive the gains; resolution scaling alone leaves
mAP@0.5 flat.
TABLE II
PER-CLASSRESCUE: AUTO-LABELS(V2)VS. HAND-LABELS(V3),
MAP@0.5:0.95
Class v2 v3
Double Statue of Ramesses II 0.000 0.995
Overseer Amenemhat 0.000 0.940
Thutmose III Statue 0.000 0.941
Naktmin and Tiy 0.269 0.977
Statue of King Akhenaten 0.281 0.955
Seated Statue of Thutmose III 0.440 0.862
labels produce sharp confidence distributions: real artifacts
cluster high while off-target scenes fall well below the rejec-
tion threshold, enabling the raised per-class thresholds (global
0.75; 0.90 for the false-positive-prone Siptah Stela).
V. BILINGUALRAG MUSEUMGUIDE
A. Grounding Principles
A public museum guide must not invent history. Three
grounding principles govern the design: (i) answers are con-
strained to retrieved museum context; (ii) retrieval is language-
matched so an Arabic question receives Arabic context; (iii)
when relevant facts are absent, the model is instructed to say
so rather than fabricate. Running inference locally removes
per-request cost, rate limits, and visitor-query leakage.
B. Knowledge Base
The knowledge base comprises 108 records covering 54
unique artifacts, each documented in English and Arabic.
A preprocessing step fixes data-quality issues — notably
a location field that had absorbed a material descriptor —
and preserves the original. Each record is split into up to
three semantically distinct chunks: anidentitychunk (name,
material, location, period), adescriptionchunk (appearance
and iconography), and asignificancechunk (historical and
cultural context). The 108 records yield215 chunks(109
English, 106 Arabic). English and Arabic chunks are stored
inseparateChromaDB collections (gem_enandgem_ar);
a question is routed only to the matching collection.C. Retrieval Strategy
Every request begins with language detection: more than
30% Arabic-script characters triggers Arabic routing, other-
wise English. Incamera mode, the English title of the detected
artifact triggers a deterministic metadata lookup returning
exactly that artifact’s chunks. Intext mode, a keyword-based
query classifier assigns the question to one of five categories
(Table III).
TABLE III
TEXT-MODEQUERYCATEGORIES ANDRETRIEVALSTRATEGY
Category Retrieval strategy
Location Metadata filter onwas_found_atfield (all matching
chunks).
Period Metadata filter on historical-overview field.
Material Metadata filter on material field.
Royal Similarity search (k= 8) against language-matched
collection.
Generic Similarity search (k= 8) against language-matched
collection.
D. Model Selection
Seven candidate models were benchmarked against 20
standardized bilingual questions. Table IV summarizes the
outcomes.Gemma 4 E2B (Q4 KM)was selected: it was
the only model producing clean, non-hallucinated answers in
bothEnglish and Arabic at the lowest cold-start VRAM.
TABLE IV
CANDIDATELANGUAGEMODELS ANDBENCHMARKINGOUTCOME
Model Quant. VRAM (MB) Verdict
Gemma 3n E2B default 1349 Eliminated (hallucination)
Gemma 3n E4B Q3 KM 2145 Finalist
Gemma 4 E2B default 2021 Eliminated (truncation)
Gemma 4 E2B Q4 KM 1389–3585 Selected
Llama 3.2 3B default 2565 Finalist (Arabic unusable)
Qwen 3 4B default 2907 Eliminated (thinking-mode)
Qwen 3.5 4B default 3485 Eliminated (latency/VRAM)
E. Pipeline Optimizations
Ten targeted optimizations reduced end-to-end latency from
over 30 s to approximately 10 s while improving answer
quality. Key changes included: hierarchical chunking (re-
placing flat documents with semantically distinct chunks),
language-separated collections, the query classifier with meta-
data filtering, mode-specific token limits (num_predict
750–1100 camera / 900–1200 text), Flash Attention (˜10–
15% faster inference),keep_alive=−1(permanent GPU
residency), a raised context window (1024→4096 tokens),
repeat_penaltyraised from 1.1 to 1.3,top_p= 0.9,
and a complete 15-royal Arabic name reference in the system
prompt to eliminate pharaoh name confusions. Fig. 4 plots the
latency progression.

BaselineMetadata FixAll OptWarm Cache0102030>30 30.2
22.7
10.2Avg. response time (s)
Fig. 4. End-to-end RAG latency across optimization phases (average over
benchmark questions): from over 30 s to a production-ready∼10.2 s.
F . RAG Evaluation Results
On a 30-question evaluation set spanning text mode, cam-
era mode, and out-of-dataset queries in both languages, the
pipeline averaged approximately 5.9 s per answer and rated
correct on all 30 responses, with no hallucinations, no trun-
cations, and no retrieval misses. All eight English text-mode,
six Arabic text-mode, six English camera-mode, three Ara-
bic camera-mode, and three vague camera-mode questions
returned complete and accurate grounded answers. All four
out-of-dataset queries (Rosetta Stone, Old Kingdom artifacts,
Great Pyramid construction, and the Rosetta Stone in Arabic)
were correctly refused without fabrication. Arabic answers av-
eraged roughly 1 s slower than English, consistent with Arabic
tokenizing to approximately twice the length of equivalent
English text.
VI. MARKETCONTEXT ANDPOSITIONING
Several tools operate in the museum-technology market, yet
none fully meets GEM’s requirements. Table V summarizes
the competitive landscape.
TABLE V
COMPETITORLANDSCAPE FORMUSEUMGUIDETECHNOLOGY
Solution Gap relative to GEM requirements
Google Arts & Culture No real-time camera recognition; no of-
fline use
Smartify Requires connectivity; not tuned for Egyp-
tian artifacts
QR-code systems Manual scanning interrupts visitor flow;
no AI narration
Audio guides No interactivity, personalization, or AR
overlay
The recurring gaps are: no real-time AI artifact recognition,
limited or no offline functionality, weak Arabic support, and no
GEM-specific experience [12], [13]. TimeLens fills these gaps
with an offline on-device detector (5.97 MB TFLite), a bilin-
gual grounded RAG guide, and a GEM-specific knowledge
base — positioning itself between heavy cloud-AI services
and simplistic QR-content systems [2].VII. DISCUSSION
Label quality is first-order.The controlled iteration study
isolates label quality as the decisive factor for fine-grained
museum artifact detection: a labeling change alone (+0.116
mAP@0.5) outweighs resolution scaling (0.000 mAP@0.5
gain). Every previously failing class was rescued by hand
annotation, confirming that architectural choices are second-
order once the label ceiling is removed.
Grounding, not model size, prevents hallucination.
The decisive RAG gains came from hierarchical chunking,
language-separated collections, and query classification — not
from a larger model. Gemma 4 E2B (Q4 KM) at 1389–
3585 MB VRAM delivers hallucination-free bilingual answers
where a 2565 MB Llama 3.2 3B produces garbled Arabic.
Limitations.The knowledge base is closed: questions out-
side 108 records receive limited answers. The query classifier
is keyword-based and can misroute questions that avoid ex-
pected vocabulary. The model has no memory across requests.
On the detection side, validation metrics reflect an image-level
split; a video-level (chronological) split would give a more
conservative generalization estimate.
VIII. CONCLUSION
This paper presented TimeLens, an AI-powered bilingual
mobile guide for the Grand Egyptian Museum comprising two
integrated subsystems. The on-device detector demonstrates
that, for fine-grained museum artifacts, label quality is the
primary determinant of recognition performance: a compact
YOLOv8n model trained on hand-verified labels achieves
mAP@0.5 = 0.995 and mAP@0.5:0.95 = 0.924 with zero
broken classes, fitting in a 5.97 MB TFLite asset. The bilingual
RAG guide — ChromaDB retrieval grounding a quantized
Gemma 4 E2B model behind FastAPI — delivers factual
English/Arabic answers at approximately 10 s end-to-end
latency with no hallucinations across a 30-question evaluation.
Both subsystems run on commodity hardware (a mid-range
phone and a single GPU) and are integrated in a production
Flutter application with bilingual interface and text-to-speech
support.
Future work includes expanding the knowledge base and
artifact dataset toward full-GEM coverage, replacing the key-
word query classifier with a learned intent model, adding con-
versational memory, and conducting formal System Usability
Scale studies in live museum conditions.
REFERENCES
[1] G. Abdelhady, A. Atef, and A. Mostafa, “Enhancing museum experi-
ences with augmented reality and machine learning: A case study of
Egyptian cultural heritage,”International Journal of Technology and
Educational Computing, vol. 4, no. 10, pp. 37–76, 2025.
[2] M. Ipalakova, Z. Bolatov, Y . Daineko, R. Sharshova, K. Abdugapparova,
and D. Tsoy, “Comparative evaluation of machine learning models
for museum exhibit recognition from video-derived datasets,”PeerJ
Computer Science, vol. 11, p. e3207, 2025.
[3] J. Wen and B. Ma, “Enhancing museum experience through deep
learning and multimedia technology,”Heliyon, vol. 10, no. 12, 2024.

[4] M. M. Alhazmi, A. Z. Alzaylaee, R. K. Qarout, and A. O. Alshutayri,
“Enhancing smart tourism: EfficientNet-based recognition of Saudi
Arabia’s cultural landmarks,”IEEE Access, vol. 13, pp. 195349–195361,
2025.
[5] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” 2023.
[Online]. Available: https://github.com/ultralytics/ultralytics
[6] T. Cheng, L. Song, Y . Ge, W. Liu, X. Wang, and Y . Shan, “YOLO-World:
Real-time open-vocabulary object detection,” inProc. IEEE/CVF CVPR,
2024.
[7] P. Lewis, E. Perez, A. Piktus,et al., “Retrieval-augmented generation
for knowledge-intensive NLP tasks,”Advances in Neural Information
Processing Systems (NeurIPS), vol. 33, pp. 9459–9474, 2020.
[8] M. Qorbani, K. Paynabar, and M. Moghaddam, “Teaching LLMs to
see and guide: Context-aware real-time assistance in augmented reality,”
arXiv preprint arXiv:2511.00730, 2025.
[9] Ollama, “Ollama: Run large language models locally,” 2024. [Online].
Available: https://ollama.com
[10] T. Liu, “YOLOv8-CDD: A salient target detection model for underwater
cultural heritage in complex environments,”International Core Journal
of Engineering, vol. 11, no. 5, pp. 332–342, 2025.
[11] Z. Dai, X. Hu, C. Chen, and H. Yu, “Occlusion handling algorithm based
on contour detection,”Journal of Advanced Computational Intelligence
and Intelligent Informatics, vol. 28, no. 4, pp. 893–900, 2024.
[12] H. Cho, “Augmenting heritage: Youth-driven AR innovation in mu-
seum spaces,”The International Archives of the Photogrammetry, Re-
mote Sensing and Spatial Information Sciences, vol. XLVIII-M-9-2025,
pp. 293–298, 2025.
[13] L. J. Choi, P. G. Cheng, and R. U. Khan, “Immersive technologies in
museums: A scoping review of cognitive outcomes related to visitor
attention, engagement, and learning,”Pakistan Journal of Life and Social
Sciences, vol. 23, no. 2, pp. 168–180, 2025.
[14] PwC, “Seeing is believing: How VR and AR will transform
business and the economy,” 2019. [Online]. Available:
https://www.pwc.ch/en/insights/digital/seeing-is-believing.html
[15] UN Tourism (UNWTO), “International tourism highlights,
2024 edition,” 2024. [Online]. Available: https://www.e-
unwto.org/doi/10.18111/9789284425808
[16] TGM Research, “Egypt travel insights 2025,” 2025. [Online]. Available:
https://tgmresearch.com