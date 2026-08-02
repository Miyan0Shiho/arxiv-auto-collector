# Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG

**Authors**: Mohamed Amine Janati, Laurent Gautier, Stéphane Barbot

**Published**: 2026-07-27 11:57:55

**PDF URL**: [https://arxiv.org/pdf/2607.24313v1](https://arxiv.org/pdf/2607.24313v1)

## Abstract
Marine life monitoring is limited by strict energy constraints, poor underwater connectivity, and the high cost of transmitting raw multimodal data from remote deployments. This paper proposes a low-consumption underwater monitoring architecture that combines always-on edge sensing with selective high-performance local reasoning. The system follows a hierarchical master--satellite design in which ultra-low-power MAX78000/MAX78002 microcontrollers continuously monitor visual and acoustic signals, while an NVIDIA Jetson Orin NX is activated only for scheduled processing, event-driven analysis, or researcher interaction. Once active, the Jetson executes a fully local multimodal pipeline for data ingestion, visual target extraction, embedding-based indexing, species identification, retrieval-augmented reasoning, and automated reporting. BioCLIP/OpenCLIP embeddings are used to organize mission data, marine taxonomic references, scientific documents, and operational metadata in local ChromaDB collections. A dedicated identification layer combines visual similarity search, centroid-based classification, and supervised classifiers to support adaptive species recognition. A LangChain-based multi-agent framework coordinates query routing, structured analysis, energy management, hardware reconfiguration, and report generation. The architecture is evaluated through visual and acoustic monitoring case studies. The proposed system bridges ultra-low-power continuous sensing with local multimodal intelligence, enabling underwater stations to produce structured, researcher-ready knowledge while compressing local data for flexible acoustic, optical, or satellite transmission, minimizing both energy use and communication overhead.

## Full Text


<!-- PDF content starts -->

Energy Constrained Hierarchical Underwater Monitoring via Local
Multi-Agent RAG⋆
Mohamed Amine Janatia,b, Laurent Gautiera,∗and Stéphane Barbota
aIfremer, 1625 RTE de Sainte-Anne, Plouzané, 29280, France
bIMT Atlantique, Technopôle Brest-Iroise, CS 83818, Plouzané, 29280, France
ARTICLE INFO
Keywords:
Ocean observation
Local AI
Agentic AI
Retrieval-AugmentedGeneration(RAG)
Species classification
Embedded AIABSTRACT
Marine life monitoring is limited by strict energy constraints, poor underwater connectivity, and
the high cost of transmitting raw multimodal data from remote deployments. This paper proposes
a low-consumption underwater monitoring architecture that combines always-on edge sensing with
selectivehigh-performancelocalreasoning.Thesystemfollowsahierarchicalmaster–satellitedesign
inwhichultra-low-powerMAX78000/MAX78002microcontrollerscontinuouslymonitorvisualand
acousticsignals,whileanNVIDIAJetsonOrinNXisactivatedonlyforscheduledprocessing,event-
driven analysis, or researcher interaction. Once active, the Jetson executes a fully local multimodal
pipelinefordataingestion,visualtargetextraction,embedding-basedindexing,speciesidentification,
retrieval-augmentedreasoning,andautomatedreporting.BioCLIP/OpenCLIPembeddingsareusedto
organize mission data, marine taxonomic references, scientific documents, and operational metadata
in local ChromaDB collections. A dedicated identification layer combines visual similarity search,
centroid-based classification, and supervised classifiers to support adaptive species recognition.
A LangChain-based multi-agent framework coordinates query routing, structured analysis, energy
management, hardware reconfiguration, and report generation. The architecture is evaluated through
visualandacousticmonitoringcasestudies.Theproposedsystembridgesultra-low-powercontinuous
sensing with local multimodal intelligence, enabling underwater stations to produce structured,
researcher-ready knowledge while compressing local data for flexible acoustic, optical, or satellite
transmission—minimizing both energy use and communication overhead.
1. Introduction
Marine ecosystems are increasingly monitored using
fixedandmobileobservationplatformsequippedwithcam-
eras,hydrophonessensors,andenvironmentalprobes.These
systems generate large volumes of heterogeneous data that
are essential for studying biodiversity, detecting ecological
changes, and identifying unusual biological or geophysi-
cal events. However, underwater deployments remain con-
strained by limited energy availability, restricted communi-
cationbandwidth,difficultphysicalaccess,andthehighcost
of transferring raw multimodal data from remote sites. As a
result, many monitoring workflows still rely on storing data
locally and processing them only after recovery or delayed
transmission, which limits real-time awareness, adaptive
sensing, and rapid scientific interpretation.
Recent advances in embedded artificial intelligence of-
fer new opportunities for local underwater data processing.
Lightweightneuralnetworkscanbedeployedonlow-power
microcontrollers to detect acoustic or visual events directly
atthesensorlevel.Atthesametime,morecapableedgepro-
cessors can execute multimodal models, retrieval systems,
and local language models for higher-level interpretation.
Nevertheless, a major architectural challenge remains: con-
tinuously operating high-performance processors is often
⋆This document presents the results of the internship funded by the
French Research Institute for Exploitation of the Sea (Ifremer).
∗Corresponding author
mohamed-amine.janati@imt-atlantique.net(M.A. Janati)
ORCID(s):0009-0002-5753-4567(M.A. Janati);0000-0002-1501-6609
(L. Gautier);0000-0002-9051-6996(S. Barbot)incompatible with long-duration underwater deployments,
while ultra-low-power detectors alone cannot provide rich
scientific reasoning, taxonomic identification, or contextual
reporting. A practical monitoring station therefore requires
a hierarchical design that combines continuous low-power
sensing with selective activation of high-capacity local in-
telligence.
This paper proposes a low-consumption, multimodal,
andagenticarchitectureforunderwatermonitoring,destined
to operate autonomously on battery for severals months.
Thesystemfollowsamaster–satellitetopologyinwhichtwo
ultra-low-power microcontrollers act as always-on sentinel
nodes, while an NVIDIA Jetson Orin NX acts as the high-
performance master node. One MAX78002 is dedicated to
visualeventdetection,whileanotherMAX78000microcon-
troller is used for acoustic monitoring. The sentinel layer
performscontinuousfirst-stagedetection,whereastheJetson
is activated only when scheduled processing, event-driven
analysis, or researcher interaction is required. This design
reduces unnecessary energy consumption while preserving
the ability to perform deeper multimodal analysis at the
edge.
The proposed system supports three operating modes.
In Autonomous Mode, the Jetson remains powered off and
wakes periodically, for example once per day, to ingest and
processdetectionsaccumulatedbythesentinelnodes.InHy-
bridMode,relevanteventsdetectedbythevisualoracoustic
sentinelstriggerimmediateJetsonactivationthroughahard-
ware wake-up interface, enabling lower-latency analysis. In
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 1 of 25
arXiv:2607.24313v1  [cs.IR]  27 Jul 2026

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Mission Mode, the Jetson remains available for direct inter-
action with researchers through a local interface, allowing
queries over recent detections, indexed mission data, and
taxonomicreferences.Thesemodesallowthesamearchitec-
ture to support long-term autonomous deployments, event-
driven monitoring, and field-based scientific inspection.
Once activated, the Jetson executes a fully local mul-
timodal processing pipeline. Collected data are ingested,
normalized,andindexedintolocalvectordatabase.Images,
video frames, scientific documents are indexed using an
embedding model, in our case OpenCLIP or BioCLIP-2.
For visual data, object localization can be performed using
GroundingDINOorYoloWorldtoextractcandidatemarine
targetsbeforeembedding.Adedicatedspeciesidentification
layerthenoperatesintheembeddingspaceusingtaxonomic
centroidsandsupervisedclassifiers,suchasSupportVector
Machines, constructed from taxonomic reference datasets.
This layer converts visual observations into taxonomic hy-
pothesesthatcanbeusedforretrieval-augmentedreasoning.
Beyond classical retrieval-augmented generation, the
system includes a local multi-agent control framework. A
RouterAgent selects the appropriate collection, modality,
and retrieval pathway for each user query or system trig-
ger. An AnalystAgent handles structured questions over
video metadata using sandboxed Pandas-based computa-
tion,avoidingunnecessarylanguage-modelgenerationwhen
deterministic analysis is sufficient. An Energy and Sensor
Management Agent adapts sensing and compute behavior
according to mission context and power constraints. A
Hardware Configuration Agent selects specialized model
binariesforthemicrocontrollersentinels,whileaReporting
Agent produces compact Markdown summaries of indexed
assets,detections,anomalies,energystate,andmodelstatus.
These reports can be stored locally and transmitted later
when connectivity becomes available.
The main contributions of this paper are as follows:
•A hierarchical energy-aware underwater moni-
toring architecturethat combines MAX78000 &
MAX78002 low-power sentinel nodes with a Jetson
Orin NX master node for selective high-performance
processing.
•A fully local multimodal retrieval and reasoning
pipelinethat indexes mission data, images, videos,
scientificdocuments,taxonomicreferences,andoper-
ationalmetadatausingBioCLIP-2/OpenCLIPembed-
dings and local ChromaDB collections.
•A dedicated species identification layerthat sup-
ports centroid-based taxonomic classification and su-
pervised classification in the multimodal embedding
space.
•A multi-agent control frameworkthat coordinates
query routing, structured analytics, energy manage-
ment, hardware reconfiguration, and autonomous sci-
entific reporting directly at the edge.•Integrated visual and acoustic monitoring work-
flowsimplemented on ultra-low-power microcon-
trollers, including visual fish detection and marine
mammal acoustic classification.
•In-situ data compression workflowtransforming
raw underwater detections into compact embeddings
for low-power transmission via acoustic, optical, or
satellite networks.
By combining ultra-low-power continuous sensing with
selective local multimodal reasoning, the proposed archi-
tecture enables underwater monitoring stations to reduce
energyconsumptionandcommunicationrequirementswhile
still producing structured, researcher-ready scientific infor-
mation at the edge.
2. Related Work
2.1. Edge AI platforms from microcontrollers to
embedded GPUs
Edge AI for monitoring spans a wide computational
spectrum, from milliwatt microcontrollers to embedded
GPU platforms. At the lowest-power end, TinyML enables
localinferenceunderseverememoryandenergyconstraints,
making it suitable for always-on sensing nodes where com-
munication or continuous high-resolution processing would
dominate the energy budget [27]. Hardware–software co-
designframeworkssuchasMCUNethaveshownthatneural
architecture search and optimized inference engines can
make convolutional models feasible on microcontrollers
with only hundreds of kilobytes of SRAM [14]. This direc-
tion is particularly relevant for autonomous marine moni-
toring, where long deployments require local preprocessing
before storing or transmitting data.
Recent ultra-low-power object-detection work has fur-
ther reduced the gap between microcontrollers and larger
edge devices. TinyissimoYOLO targets object detection
on highly constrained processors and was evaluated on
platforms including the MAX78000, GAP9, STM32 and
Apollo-class low-power microcontrollers [19, 21]. Maxim
/ Analog Devices devices are especially relevant because
the MAX78000 and MAX78002 integrate a convolutional
neural-networkacceleratorwhilescoringverylowonpower
consumption benchmarks[18]. The MAX78002 extends
this family with larger CNN memory and sensor-oriented
interfacessuchasMIPICSI-2andI2S,whichareusefulfor
compact camera and acoustic front-ends [3].
Abovethemicrocontrollertier,RaspberryPi-classsingle-
board computers provide a practical compromise between
cost, programmability, and model complexity. They have
been used in smart camera-trap and edge–cloud systems
for real-time animal detection and species recognition, in-
cluding continual-learning wildlife monitoring and ant-
species detection with YOLO-based pipelines [30, 24]. For
more demanding underwater perception tasks, NVIDIA
Jetson platforms provide GPU acceleration for real-time
deep learning. For example, Jetson-based deep-sea crawler
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 2 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
deploymentshaveusedYOLOmodelsforautomatedspecies
classificationandcountinginbenthicimagery[23].Another
previous work carried by our team was the detection of
invasive species, specifically LionFish, using Rasberry Pi
modulewithaNPUaccelerator[13].Theseworksshowthat
each platform tier is useful, but they also motivate architec-
tures that combine them: an always-on microcontroller can
performinexpensiveeventfiltering,whileaRaspberryPior
Jetsoncanbeactivatedonlywhenrichervisual,acoustic,or
multimodal inference is required.
2.2. Hierarchical cascade systems and
event-triggered monitoring
A central challenge in autonomous monitoring is that
the most accurate models are often too energy-intensive
to run continuously. Hierarchical cascade systems address
this problem by assigning low-cost decisions to low-power
devices and escalating only uncertain or promising events
to more capable processors. This principle appears in early
wireless-sensor work on radio-triggered wake-up, where a
low-power front-end activates a sleeping node only when
communicationorsensingisneeded[10].Similarideashave
beenappliedtovisualmonitoring:sleepyCAM,forexample,
uses an external low-power trigger to wake or power a
RaspberryPicamerasystem,reducingthecostofcontinuous
videosurveillance[17].Suchdesignsaredirectlycompatible
with edge AI cascades, where a microcontroller first runs a
coarsedetectororthresholdmodel,thenwakesaRaspberry
Pi,Jetson,orcommunicationmoduleforexpensiveinference
and reporting.
Marine monitoring deployments are strictly limited by
battery, bandwidth, and storage, making tiered cascade pro-
cessing essential for long-term operation. As demonstrated
by real-time baleen-whale monitoring from ocean gliders
[6], a continuous, low-power first stage can screen raw
acoustic or visual streams to flag candidate events, reserv-
ing resource-intensive high-tier processing, accurate multi-
modaldetection,anddatatransmissionstrictlyfordetections
of interest.
2.3. Multimodalretrieval,vision–languagemodels,
and embedding-based species identification
Vision–language models and contrastive embedding
modelsprovideacomplementarydirectionforspeciesiden-
tification.CLIPshowedthatimageandtextencoderstrained
with contrastive learning can support zero-shot recognition
and retrieval by mapping images and natural-language la-
bels into a shared embedding space [26]. Biology-specific
extensions make this paradigm more relevant for ecological
applications. BioCLIP adapts vision foundation models
to biological taxonomy and biodiversity imagery, while
CLIBD extends contrastive learning toward the joint use of
images, DNA barcodes, and taxonomic text for biodiversity
monitoring[29,9].Thesemethodsareusefulwhenthegoal
is not only to detect an organism, but also to compare it
with a reference collection, retrieve similar examples, and
use taxonomic metadata to support identification.For marine species recognition, large curated datasets
and benchmarks are essential because fine-grained visual
differences between species are often subtle. FishNet pro-
videsalarge-scalebenchmarkforfishrecognition,detection,
and trait prediction, supporting the development of mod-
els that are better aligned with marine biodiversity tasks
than generic object-recognition datasets [12]. A practical
retrieval-basedpipelinecanthereforebebuiltbyembedding
a labeled reference dataset of fish or marine organisms and
storingtheresultingvectorsinasearchindex.Classifyinga
new observation is then performed via cosine similarity or
𝑘-nearest-neighbor search. The retrieved nearest neighbors
yieldbothapredictedlabelandinterpretablesupportingev-
idence,suchasvisuallysimilarreferenceimages,taxonomic
names, habitat metadata, or confidence estimates.
Multimodalretrieval-augmentedgenerationextendsthis
idea by combining retrieval with a generative or reasoning
model.Insuchasystem,image,textoracousticembeddings
retrieve relevant reference examples and metadata before a
vision–language or multimodal language model produces a
final explanation or report. Recent fisheries-oriented multi-
modal RAG work follows this direction by integrating het-
erogeneousfisherydatasourcesfordomain-specificretrieval
andreasoning[28].Forautonomousmarinemonitoring,this
suggests a sequential approach : Filtering detections with
low-consumption models, then generating species hypothe-
ses and answers with LLM models.
3. Datasets and Taxonomic Resources
This section groups all data sources used by the system.
Thefirstpartdescribesthedatasetsusedtotrainandvalidate
thetwolow-powersentinelsubsystems:visualfishdetection
andmarinemammalacousticclassification.Thesecondpart
describes the taxonomic reference collections used by the
high-fidelity identification and retrieval layer after Jetson
activation.
3.1. FishDet-M Image Dataset
The objective of the visual sentinel is to detect under-
water events and then decide, according to the currently
selected operating mode, whether the Jetson should be trig-
geredorjuststorethedataontheSDCard.Asanapplication
example,werestricttofishdetection.Thevisualmodelsare
trained on the FishDet-M dataset [1].
FishDet-M is a large-scale, unified benchmark dataset
designed for robust fish and underwater object detection.
To address the historical challenge of fragmented and niche
marine vision datasets, FishDet-M consolidates 13 distinct
public underwater datasets, including well-known sources
such as DeepFish, FishNet, Brackish-MOT, and TrashCan
1.0, into a single standardized framework. The resulting
collection contains 105,556 images and 296,885 annotated
fish instances, all unified under standard COCO-style anno-
tations with both bounding boxes and segmentation masks.
Thedatasetisexplicitlycuratedtotestdomaingeneralization
across highly diverse aquatic visual domains, ranging from
tropical coral reefs and marine environments to brackish
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 3 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Table 1
Overview of the FishDet-M dataset splits and annotations
before and after restricting to five or fewer fish instances per
image.
Feature Training Set Validation Set Test Set
Original Dataset (No Restrictions)
Number of Images 83,093 10,654 11,809
Number of Fish Instances 228,558 32,961 35,366
Filtered Dataset (≤5 Fish Instances per Image)
Number of Images 76,202 9,626 10,645
Number of Fish Instances 95,199 12,290 14,060
waters, aquaculture tanks, and indoor aquariums. As a re-
sult, models trained on this dataset are exposed to realistic
deployment challenges such as high turbidity, severe occlu-
sions, motion blur, muted contrast, and green-blue spectral
distortion.
Toenhancemodelgeneralizationandpreventoverfitting,
a dynamic stochastic data augmentation strategy is used
during training. For each sample, the pipeline dynamically
generates two distinct versions by randomly applying trans-
formations such as geometric modifications, color jittering,
and noise insertion. We restricted the training and testing
phases to images containing five or fewer bounding boxes.
First, the models used are too compact to capture very
fine details. Second, resizing the images to 256×320 pixels
causes some bounding boxes to become excessively small
andirrelevant.Finally,accordingtoTinyissimoYolomodel
original paper, the authors gained approximately 20% mAP
bylimitingdetectiontoamaximumoffiveobjectsperframe
[20]. The final dataset repartition is resumed in table 1.
3.2. Marine Mammal Acoustic Dataset
Fortheinitialunderwateracousticexperiments,thesys-
tem focuses on marine mammal detection. Many marine
mammals emit sounds in the frequency range from 200
Hz to 8,000 Hz, which aligns with the memory limitations
of the MAX78000 microcontroller. To natively match the
hardwarecapturebehavioroftheMAX9867audioCODEC,
all input audio is isolated to the left channel, resampled to
a 16 kHz sample rate, and divided into 0.98-second frame
windows.
TheprimarydatasetselectedforthistaskistheWatkins
Marine Mammal Sound Database [31]. This open-access
historicalarchivecontainsthousandsofdigitizedunderwater
audio recordings of more than 60 species of whales, dol-
phins, and seals, collected globally over seven decades. To
prevent class imbalance from skewing model performance,
the dataset is restricted to the 12 most heavily represented
species.
To make the model robust against ambient underwa-
ter sounds, aNoiseclass is introduced as Class 0. The
datasetusedtorepresentthisbackgroundnoiseisDeepShip
[11],abenchmarkunderwateracousticsdatasetconsistingof
more than 47 hours of real-world passive sonar recordings.DeepShipiscommonlyusedtotrainmachinelearningmod-
els that classify four distinct commercial ship types: cargo,
passenger, tanker, and tug.
Table 2
Dataset distribution across classes. Number of elements
and durations for Whale and Dolphin classes are sourced
from the WHOI Dataset data exploration. Noise class is
sourced from DeepShip.
Class Name Audio files Duration (h)
Noise (DeepShip) 63 > 47.0
Common Dolphin 884 0.64
False KillerWhale 508 0.27
FinbackWhale 580 3.90
HumpbackWhale 604 1.99
KillerWhale 2647 1.65
Long Finned PilotWhale 1104 0.86
Pantropical Spotted Dolphin 1025 0.85
Short Finned PilotWhale 607 0.39
SpermWhale 1379 12.26
Spinner Dolphin 524 0.66
Striped Dolphin 681 0.36
White sided Dolphin 560 0.45
The inherent static noise produced by the MAX9867
audio CODEC on the MAX78000 platform creates a third
technical challenge. To address this issue and ensure the
Noiseclass matches the volume of the most heavily repre-
sentedspecieswithoutoverfitting,thedatasetisdynamically
balancedusingrealhardwarebaselinerecordingscombined
with a three-pronged synthetic noise generation strategy:
1.Board Noise Crops:Random 0.98-second windows
areextractedfromrealMAX9867backgroundrecord-
ings and perturbed with Gaussian jitter (𝜎= 2on a
16-bit scale) to simulate varied board idle floors.
2.White Gaussian Noise:Flat-spectrum noise gener-
atedatrandomamplitudesreflectingthetypicalboard
idle floor (-100 dBFS to -60 dBFS).
3.Pink (1/f) Noise:Noise that rolls off at 3 dB/octave,
generated between -90 dBFS and -55 dBFS, closely
mimicking the ambient ocean spectral shape found in
real underwater recordings.
Prior to Mel spectrogram generation, all data (both real
and synthetic) is scaled to a 16-bit PCM integer range
and passed through an exact 1st-order IIR high-pass filter
(coefficients𝑏=[1.0,−1.0],𝑎=[1.0,−0.995]) to perfectly
mirror the C-level firmware execution on the hardware.
The aggregated dataset is ultimately partitioned into a
70% training, 20% validation, and 10% testing split. By
handlinghardwarenoiseandsignalfilteringduringthetrain-
ing phase, the system avoids the need for complex and
computationally expensive noise-filtering steps during real-
timepreprocessing.Thisminimizesbothlatencyandenergy
consumption. After merging all datasets, a standardized
pipelineisappliedtogenerate96x64log-Melspectrograms.
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 4 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Figure 1:Acoustic data generation pipeline.
Thispipelineisstrictlydesignedtoremainmemoryefficient,
because the same preprocessing steps must be executed
locallyontheMAX78000inrealtime,whileavoidingbiases
thatcouldleadtomisclassification.Figure1summarizesthe
complete acoustic dataset generation process.
3.3. Taxonomic Reference Collections and
Ingestion
The high-fidelity identification layer relies on extensive
taxonomic reference collections to cross-reference sensor
observations. As summarized in Table 3, the project lever-
agesfiveindexedreferencedatasets—FishBase/SeaLifeBase
[8][25], BioTrove[32], FishNet [22], Fish-Vista[16], and
WildFish++[33]—encompassing493,387visualentriesin
total.OzFish[5]isintentionallyexcludedfromthereference
index and reserved strictly as an external evaluation set.
Table 3 summarizes the different taxonomic reference
datasets by specifying the number of images, species, gen-
era,families,orders,andclasses,aswellastheirgeographic
or ecological scope.
4. System Architecture
This section describes the proposed underwater moni-
toringsystemarchitecture.Thesystemisdesignedarounda
hierarchical edge-computing structure in which low-power
sentinelmicrocontrollerscontinuouslymonitortheenviron-
ment, while an NVIDIA Jetson Orin NX is activated only
when higher-capacity multimodal processing is required.
The section first introduces the hardware hierarchy and the
power-aware operating modes. It then describes the local
multimodalprocessingpipelineexecutedontheJetsonOrin
NX, including detection-aware ingestion, visual target ex-
traction, multimodal embedding, species identification, and
retrieval-augmentedreasoning.Finally,itpresentsthemulti-
agent control framework responsible for routing, analysis,
reporting, energy management, and hardware adaptation.
4.1. Hierarchical Structural Overview
The proposed underwater monitoring system follows a
hierarchical master–satellite architecture designed to recon-
cilecontinuousenvironmentalmonitoringwithstrictenergy
constraints.Thearchitectureseparatesalways-onlow-power
perception from high-performance multimodal reasoning.At the lower tier, a MAX78000 microcontroller and
another MAX78002 microcontroller operate as low-power
sentinel nodes dedicated to acoustic and visual monitoring.
These subsystems continuously execute lightweight neural-
network models to detect potentially relevant events, such
as marine mammal vocalizations or visual activity. Their
role is not to perform complete scientific interpretation,
but to provide an energy-efficient first filtering stage that
determines whether an event deserves deeper analysis.
Attheuppertier,anNVIDIAJetsonOrinNXactsasthe
main computational hub. It is responsible for high-fidelity
inference, multimodal embedding, local vector indexing,
species identification, retrieval-augmented generation, re-
searcher interaction, reporting, and agentic orchestration.
Undernormalconditions,theJetsonremainsshutdownand
is activated only when scheduled processing, event-driven
analysis, or researcher interaction is required.
This division of labor allows the system to preserve
energy duringlong-duration underwater deploymentswhile
stillprovidingaccesstocomputationallyintensiveAIpipelines
when the collected data justify deeper analysis.
Figure 2 resumes all the points mentionned in this sec-
tion. Figure 3 shows a prototype of the developed system.
4.2. Power-Aware Operation and Hardware
Orchestration
Thesystemsupportsthreeoperatingmodes,eachcorre-
spondingtoadifferenttrade-offbetweenautonomy,latency,
powerconsumption,andresearcherinteraction.Inallmodes,
thelow-powersentineltierremainsresponsibleforcontinu-
ous environmental monitoring, while the Jetson Orin NX is
activated selectively.
4.2.1. Autonomous Batch Mode
Autonomous Batch Mode is intended for long-duration
deployments where energy preservation is the main con-
straint. The MAX-based acoustic and visual subsystems
remain active and locally store raw detections on their re-
spective SD cards.
In this mode, the Jetson Orin NX remains turned off
most of the time and wakes according to a scheduled cycle,
for example once per day. During this scheduled activation,
the Jetson ingests the accumulated detections, transfers the
correspondingfilesfromthesentinelstoragearraysthrough
UART,performssecondaryvalidationusinghigher-capacity
models, indexes the new data, generates compact scientific
summaries, and then returns to a low-power state.
4.2.2. Event-Driven Hybrid Mode
Event-Driven Hybrid Mode prioritizes low-latency sit-
uational awareness while preserving the sentinel-first archi-
tecture. When a low-power subsystem detects an event of
interest, it immediately wakes the Jetson Orin NX through
thehardwareorchestrationlayer.TheJetsoncanthenretrieve
the corresponding raw data, activate additional sensors if
necessary, perform deeper multimodal analysis, and update
the local database shortly after the event occurs.
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 5 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Table 3
Taxonomic image collections used by the project. Image counts correspond to indexed visual entries when the collection is used
as a reference. OzFish is reported separately because it is used only for evaluation. Taxonomic counts are computed after the
project normalization step and, when available, completed with the local taxonomy cache.
Dataset Images Species Genera Families Orders Classes Image conditions
FishBase /
SeaLifeBase 49,149 17,379 5,489 1,436 213 60 Reference photographs from curated databases;
mixed live, aquarium, field, specimen, and pre-
served/dead photographs depending on source
metadata.
BioTrove 190,534 19,586 5,739 1,074 188 35 Heterogeneous reference images organized by taxo-
nomic class and species; conditions are mixed rather
than exclusively in situ.
FishNet 94,349 17,610 3,973 581 81 10 Mostly object/reference-style fish images, including
multiple life stages and non-field compositions; not
a pure underwater in situ dataset.
Fish-Vista 56,321 729 309 120 47 3 Curated fish photographs from GLIN, iDigBio, and
MorphBank;mixedspecimen,museum,andfield-like
images rather than only live underwater scenes.
WildFish++ 103,034 2,348 931 255 62 7 Real-world wild fish imagery from train and valida-
tion folders, closer to natural visual conditions than
the database reference collections.
OzFish 1,903 198 84 29 10 2 Real fish observations with species boxes and mea-
surement files; used to test cross-source generaliza-
tion.
Figure 2:Hierarchical, low-power underwater monitoring architecture. Edge-level microcontrollers (MAX78000/02) continuously
monitor environmental events and selectively wake a high-performance master node (Jetson Orin NX) for local multimodal
processing, retrieval-augmented generation, and researcher interaction.
Figure 3:Experimental implementation of the system
Thismodeissuitablefordeploymentsinwhichrelevant
biological or environmental events must be analyzed soonafter detection, while continuous operation of the high-
power compute module remains undesirable.
To implement this behavior, the event detection sets
microcontrollers’ I/O that are connected to a sharedTTL
wake line, as illustrated in Fig. 4. This simple diode-OR
configuration allows any microcontroller to initiate a wake-
up event while preventing current from flowing between
GPIO outputs. Assertion of the shared event line activates
the hardware power-switching stage, which connects the
12Vsupply to the Jetson.
After booting, the Jetson asserts a dedicated GPIO con-
nectedtothesamecontrolnodethroughanadditionalisola-
tion diode. This signal provides a self-hold function, main-
taining power after the initiating event signal has been re-
leased. Once event processing, data transfer, and database
updates are complete, the Jetson performs an orderly soft-
ware shutdown and releases the hold signal, causing the
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 6 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Figure 4:Hardware wake-up and power-hold circuit used in
Event-Driven Hybrid Mode.
power-switching stage to disconnect the12Vsupply. Con-
sequently, short detection pulses are sufficient to initiate a
complete processing cycle without requiring the detecting
microcontrollertoremainactivethroughouttheJetsonoper-
ating period.
4.2.3. Mission and Researcher-Interactive Mode
MissionandResearcher-InteractiveModeisdesignedfor
field operations in which researchers are physically present
near the monitoring station and require direct access to the
localintelligencelayer.Inthismode,theJetsonOrinNXre-
mainsactiveandexposesalocalwebinterfaceimplemented
using HTML, CSS, and JavaScript. Unlike Autonomous
Batch Mode and Event-Driven Hybrid Mode, this mode
prioritizes interactive exploration, rapid scientific verifica-
tion,anddirectaccesstoindexeddataoverminimumpower
consumption.
The interface is organized into four main tabs. The
first tab provides a chatbot interface connected to the local
agentic retrieval system. Researchers can submit natural-
language queries, which are processed by the backend
through the local multimodal RAG pipeline and the unified
agenticreasoninglayer.Dependingonthequery,thesystem
routestherequesttowardtheappropriateChromaDBcollec-
tion, retrieval modality, analytical pathway, or local instruct
model.
The second tab is dedicated to species identification. A
researchercanuploadanimageofanunknownorganismand
select the reference collection to be used for identification.
The backend then compares the image embedding against
precomputedtaxonomiccentroidsandSVMclassifiers.The
interfacereturnsthemostlikelyspecies,genus,family,order,
and class, together with the corresponding confidence indi-
cators.ForSVM-basedpredictions,thereportedvaluecorre-
spondstotheclassifierprobability,whileforcentroid-based
predictions it corresponds to the cosine similarity between
the query embedding and the selected taxonomic centroid.
Figure 5 shows a specie identification example, the Web
UI displays the five taxonomic levels (class, order, genus,
family and specie). The third tab supports visual similarity
Figure 5:Specie identification web interface
search. The researcher can upload or select a query image,
describe a text query, choose a target collection, and define
the number of nearest neighbors𝑘to retrieve. The search
can be performed either over marine taxonomic reference
datasets or over the local multimodal mission collection.
Theinterfacethendisplaysthetop-𝑘closestimagestogether
withtheirsimilarityscores,allowingresearcherstovisually
inspectrelatedexamples,compareuncertaindetections,and
validate retrieval results. Figure 6 shows a similarity search
onimage,andfigure7showsasimilaritysearchbutthistime
on text.
The fourth tab provides a database overview. It lists
the currently available ChromaDB collections and reports
the number of indexed items in each collection. This view
allows researchers to verify the state of the local memory,
inspectwhethernewdetectionsorreferenceassetshavebeen
indexed, and monitor the growth of the onboard database
during field operations.
Overall,MissionandResearcher-InteractiveModeturns
themonitoringstationintoalocalscientificinspectionplat-
form. It allows researchers to query the RAG system, iden-
tify species from uploaded images, retrieve visually similar
examples, and inspect the current database state directly at
the edge without requiring cloud connectivity.
4.3. Local Multimodal Processing Pipeline
OncetheJetsonOrinNXisactivated,thecollecteddata
areprocessedthroughafullylocalmultimodalpipeline.This
pipelinetransformsrawdetectionsintoindexed,searchable,
and interpretable scientific information. It links real-time
detections with historical mission data, taxonomic refer-
ences,andscientificdocumentationwithoutrequiringcloud
connectivity.
The pipeline consists of four main stages: (i) detection-
aware data ingestion, (ii) target localization, (iii) species
identification, multimodal embedding, and indexing, and
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 7 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Figure 6:Example of image-image similarity search using the Web UI
Figure 7:Example of text-image similarity search
(iv) retrieval-augmented reasoning. This modular organiza-
tionclearlyseparatesdatamanagementfromspeciesidenti-
fication, making the role of each stage explicit and facilitat-
ing future extensions.
Before the pipeline can be executed, the species identi-
fication layer must be initialized. During this configuration
stage,asetofreferencetaxonomicdatasetsisindexed,after
which class centroids and Support Vector Machine (SVM)classifiers are computed. These classifiers are subsequently
used by the online multimodal pipeline to generate species
hypotheses. The species identification layer is described in
detail in Section 4.3.4.
The overall workflow is as follows. An event detected
by the Sentinel system triggers the acquisition process. De-
pending on the operating mode, the acquired data (images
or audio recordings) are either stored on the SD card for
subsequent processing or captured directly after the Jetson
Orin NX is awakened from its low-power state. The target
localization stage extracts the relevant regions of interest
from the acquired images using text-guided localization
prompts. These image crops are then forwarded to the em-
beddingandindexingstage,whereBIOCLIP-2embeddings
are computed. The resulting embeddings are subsequently
processedbythespeciesidentificationlayertogenerateone
or more hypotheses regarding the detected species.
The computed embeddings, together with their associ-
ated metadata—including timestamps, species hypotheses,
mission identifiers, sensor information, and video sources
are stored in ChromaDB collections. These indexed rep-
resentations form a searchable knowledge base that con-
tinuously grows throughout the mission. When a query is
submitted by either a human operator or an autonomous
agent, the system performs retrieval-augmented generation
(RAG) over the indexed ChromaDB collections. The re-
trieved embeddings and their associated metadata are pro-
videdascontextualinformationtothelargelanguagemodel
(LLM), enabling scientifically grounded responses based
on both the current observations and previously collected
mission data.
The following sections describe each stage of the pro-
posed pipeline in detail. The presentation of the species
identification layer is intentionally deferred until after the
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 8 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Figure 8:Local multimodal processing and retrieval-augmented memory pipeline.
embedding stage, as it relies on embedding representations
introduced in the preceding section.
4.3.1. Detection-Aware Data Ingestion
Thepipelinebeginswithdetectionsproducedbythelow-
power sentinel subsystems. These detections may include
acousticframes,seismicwaves,visualobservationsorasso-
ciatedmetadata.InAutonomousBatchMode,detectionsare
accumulated locally and transferred through UART to the
Jetson during scheduled wake-up cycles. In Event-Driven
Hybrid Mode, relevant detections can trigger immediate
Jetson activation and data transfer.
At this stage, the objective is not yet to identify species
or generate ecological explanations. The goal is to collect,
normalize,timestamp,andorganizethedataproducedbythe
sentinel layer so that they can be processed by the higher-
capacity multimodal pipeline. Each ingested item is associ-
atedwithmetadatasuchassourcesubsystem,timestamp,file
path, detection type, and triggering mode.
4.3.2. Target Localization and Crop Extraction
For visual inputs, the pipeline adopts a detection-first
strategy prior to vector database insertion, as illustrated in
Figure8.Theobjectiveofthisstageistoisolatesemantically
relevant marine organisms from the surrounding scene so
thatthesubsequentembeddingprocessfocusesprimarilyon
thetargetratherthanonpotentiallydominantbackgroundel-
ements,suchaswater,rocks,sand,vegetation,oracquisition
equipment.
TargetlocalizationisperformedusingGroundingDINO
[15], an open-vocabulary object detector that combines
the transformer-based DINO detection architecture with
groundedvision–languagepre-training.Unlikeconventional
closed-set detectors, which are restricted to a predefined
collection of object classes, Grounding DINO receives
natural-language expressions as detection queries. It jointly
encodestheinputimageandthetextualprompt,alignstheir
visual and linguistic representations, and returns bounding
boxes associated with the image regions that are mostsemantically consistent with the requested concepts. This
property is particularly suitable for marine imagery, for
whichthetaxonomiccategoriesofinterestmayvarybetween
deployments and may not be adequately represented by
standard object-detection datasets.
The pipeline uses the Grounding DINO Tiny variant,
based on a Swin-T visual backbone. Despite the “Tiny”
designation,thecompletemodelcontainsapproximately172
million parameters. The storage required by the parameters
alone is therefore approximately688MB when represented
in32-bitfloating-pointprecisionandapproximately344MB
under16-bitprecision.Thesevaluesrepresentonlythetheo-
retical memory required to store the model weights. During
inference, additional GPU memory is required for visual
feature maps, language representations, cross-modal atten-
tion tensors, decoder queries, intermediate activations, and
framework-level CUDA allocations. Consequently, several
gigabytes of GPU memory must be reserved in practice;
an allocation of approximately4GB provides a reason-
able operating margin for single-image inference, although
the precise peak consumption depends on the input reso-
lution, numerical precision, software implementation, and
memory-caching behavior.
At inference time, the detector is queried using coarse
semantic categories such asfish,cnidarians,mollusks, and
crustaceans. These prompts are intentionally broader than
species-level labels because the purpose of this stage is
targetlocalizationratherthanfine-grainedtaxonomicidenti-
fication. For every input frame, Grounding DINO produces
a collection of candidate bounding boxes and their corre-
spondingconfidencescores.Low-confidencepredictionsare
rejected, after which Non-Maximum Suppression (NMS) is
applied to remove strongly overlapping detections that are
likely to correspond to the same organism. The remaining
regions of interest are cropped from the original frame
and cached locally, thereby avoiding repeated detection and
crop-extraction operations during subsequent indexing or
retrieval experiments.
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 9 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
As summarized in Figure 8, the retained crops are sub-
sequently passed to the visual embedding model and in-
sertedintothevectordatabasetogetherwiththeirassociated
metadata. Performing detection before feature extraction
reduces the influence of irrelevant background content and
producesembeddingsthatmoredirectlyrepresentthevisual
characteristics of the detected organism. This is especially
important when the target occupies only a small portion of
the original image or when multiple background structures
exhibit stronger visual patterns than the organism itself.
If no candidate satisfies the detection-confidence crite-
ria, the complete frame is retained as a fallback representa-
tion. This conservative mechanism prevents potentially in-
formativeobservationsfrombeingdiscardedduetodifficult
imaging conditions, small or partially occluded organisms,
domain shifts, or imperfect agreement between the selected
textual prompts and the visible target. Such samples can
thereforeremainavailablefordownstreamidentification,re-
trieval,andanomalyanalysisevenwhenreliablelocalization
is not achieved.
Among the models involved in the indexing pipeline,
Grounding DINO is the most computationally expensive
component. Its computational cost arises from the simul-
taneous processing of high-resolution visual features and
textual representations, together with repeated cross-modal
attention and bounding-box decoding operations. In com-
parison, crop extraction, NMS, local caching, and vector-
database insertion introduce relatively limited overhead.
Grounding DINO consequently accounts for the largest
portionoftheper-frameprocessingtimeandconstitutesthe
principal computational bottleneck of the detection-aware
indexing procedure. This motivates executing localization
only once during ingestion and caching its outputs, rather
thanrepeatingdetectionwheneveranindexedobservationis
queried. The final output of this stage is a set of localized
candidate regions—or the original frame when localization
fails—that can subsequently be embedded, indexed, identi-
fied, and retrieved.
We also evaluated YOLO World [7], which achieved
very similar detection performance to Grounding DINO
while operating at substantially lower computational cost.
Specifically, YOLO World completes inference in a frac-
tion of the time required by Grounding DINO for equiv-
alent tasks, making it well-suited for real-time or near-
real-time applications where throughput is critical. Despite
this speed advantage, its semantic grounding capability and
open-vocabulary detection accuracy remain comparable to
those of Grounding DINO, ensuring that neither approach
introduces meaningful differences in downstream retrieval
quality.YoloWorldmayalsobedeployedthroughTensorRT
which makes the inference far faster specially on Jetson
devices.
4.3.3. Multimodal Embedding and Vector Indexing
Aftercandidatedatahavebeencollectedandtherelevant
visual regions have been extracted, the pipeline convertseachelementintoafixed-dimensionalnumericalrepresenta-
tion,referredtoasanembedding.Theseembeddingsencode
the semantic content of an input and provide a common
representation in which heterogeneous modalities can be
compared and retrieved efficiently.
The embedding stage utilizes Contrastive Language–
Image Pre-training (CLIP) [26], a multimodal architecture
featuring separate visual and text encoders. By mapping
matching image–text pairs into a shared embedding space
duringtraining,CLIPalignssemanticallyrelatedvisualand
textualrepresentations,enablingdirectcross-modalcompar-
ison without a task-specific classification head.
In the proposed pipeline, visual inputs—including de-
tected image crops and sampled video frames—are pro-
cessed using either BioCLIP-2 or a general-purpose Open-
CLIP visual encoder. OpenCLIP is an open-source imple-
mentation of the CLIP architecture that provides pretrained
models with different visual backbones, training datasets,
andcomputationalrequirements.Itoffersageneralrepresen-
tation of visual and linguistic concepts learned from large
collections of image–text pairs. However, because these
training collections primarily contain broad web imagery,
theresultingrepresentationmayprovidelimiteddiscrimina-
tion between visually similar biological taxa.
BioCLIP [29] addresses this limitation by adapting the
CLIP training paradigm to the biological domain. It re-
tains the dual-encoder architecture and contrastive learning
objective of CLIP but is trained using TreeOfLife-10M, a
large-scaledatasetcontainingimagesofanimals,plants,and
fungitogetherwithstructuredtaxonomiclabels.Thetextual
supervision incorporates the biological hierarchy, including
taxonomicrankssuchaskingdom,phylum,class,order,fam-
ily, genus, and species. As a result, the learned embedding
spacecapturesbothvisualsimilarityandbiologicallymean-
ingful relationships between organisms. This domain spe-
cializationmakesBioCLIPmoreappropriatethanageneral-
purpose CLIP model for representing marine species, par-
ticularly when the downstream task requires fine-grained
differentiation between morphologically similar taxa.
For each visual observation𝐼𝑖, the visual encoder pro-
duces an embedding vector
𝐯𝑖=𝑓img(𝐼𝑖),(1)
where𝑓imgdenotes the BioCLIP or OpenCLIP image en-
coder. Similarly, each textual element𝑇𝑗, such as a scien-
tific name, species description, ecological trait, document
chunk, or operational annotation, is transformed using the
corresponding text encoder:
𝐭𝑗=𝑓text(𝑇𝑗).(2)
Beforeindexing,theresultingvectorsare𝐿2-normalized:
̂𝐳=𝐳
‖𝐳‖2,(3)
where𝐳represents either a visual or textual embedding.
Semanticsimilaritybetweentwonormalizedrepresentations
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 10 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
is subsequently measured using cosine similarity, which
reduces to their dot product:
𝑠(̂𝐳𝑎,̂𝐳𝑏)=̂𝐳⊤
𝑎̂𝐳𝑏.(4)
Alargersimilarityvalueindicatesthatthecorrespondingin-
putsaremorecloselyrelatedwithinthelearnedmultimodal
space. This formulation supports image-to-image, text-to-
text,andcross-modaltext-to-imageretrievalusingthesame
indexing mechanism.
The computational cost and memory footprint of this
stagedependontheselectedCLIPbackbone.Modelsbased
on larger Vision Transformer architectures generally pro-
vide richer representations but require more parameters,
GPUmemory,andinferencetime.Intheproposedpipeline,
embedding extraction is significantly less computationally
expensivethantheprecedingGroundingDINOlocalization
stage [15] and exhibits a computational footprint highly
comparable to YOLO-World [7]. Nevertheless, processing
a large number of crops or video frames can produce a
substantial cumulative cost, as image scenes containing a
high density of detections yield a proportionally higher
number of localized crops requiring embedding generation.
Embeddings are therefore computed once during ingestion,
stored persistently, and reused during subsequent retrieval
and identification operations.
Mission images, localized video crops, textual species
descriptions, PDF chunks and operational metadata are
stored in modality-specific local ChromaDB collections.
ChromaDB provides a vector-oriented storage layer that
associates each embedding with a unique identifier, its
original content or file reference, and structured metadata.
Dependingonthedatamodality,thestoredmetadatainclude
the source-file path, frame or document index, acquisition
timestamp, detected category, bounding-box coordinates,
mission identifier, and preprocessing configurations. Cru-
cially, the metadata also encompass hypotheses regarding
taxonomic classification, which are detailed in the subse-
quent subsection. Preserving this provenance ensures that
each retrieved vector remains traceable to its corresponding
observation, thereby preventing the embedding space from
decoupling from the original mission data. When multiple
crops within a single image are assumed to belong to
the same species, only one representative crop is retained.
The total number of such instances is recorded within the
metadata of the retained crop. This mechanism prevents
the saturation of ChromaDB with redundant embeddings
of the same species (e.g., a large school of sardines) while
simultaneously preserving the original instance count.
During retrieval, a query is encoded using the appro-
priate pathway and compared with the indexed vectors.
ChromaDBreturnsthenearestcandidatesaccordingtotheir
vector distance or similarity. For example, a textual query
describing a marine organism can retrieve visually compat-
ible image crops, while an image crop can retrieve related
species descriptions or visually similar observations.
As previously noted, indexing the ingested data entails
storing hypotheses across five distinct taxonomic levels asmetadata. The methodology for generating these classifica-
tions is entirely delegated to the species identification layer,
which is detailed in the subsequent subsection.
4.3.4. Species Identification Layer
Althoughmultimodalembeddingsprovideaunifiedrep-
resentation for images and text, similarity within this space
does not inherently constitute a definitive taxonomic deci-
sion.Consequently,adedicatedspeciesidentificationlayeris
introducedpriortothedataindexinglayer.Thislayerutilizes
embeddings from reference sets to generate a ranked list of
taxonomic hypotheses for each detected crop, accompanied
by confidence scores and supporting evidence.
The architecture operates directly in the pretrained Bio-
CLIP -2 embedding space, avoiding retraining for every
missionorgeographicscope.Missionadaptationisachieved
by updating reference collections rather than maintaining
multiple separately trained CLIP models: a coastal survey
canusefish-specificreferences,whileabroaderbiodiversity
survey incorporates mollusks, cnidarians, and crustaceans
alongside fish.
Eachreferenceimageisembeddedtogetherwithataxo-
nomicallystructuredtextualrepresentation.Foranorganism
withknowntaxonomy,thetextpromptfollowsthetemplate
𝑇=“a photo of⟨class⟩ ⟨order⟩ ⟨family⟩ ⟨genus⟩ ⟨species⟩”,
(5)
where every placeholder is replaced by its authoritative
taxonomic value. The structured prompt provides the text
encoder with both fine-grained species identity and hierar-
chical context.
Becausesourcereferencecollectionsdonotalwayspro-
videallfivetaxonomicranks,missinggenus,order,orclass
values are reconstructed through a taxonomy-resolution in-
frastructure.Namenormalizationandtaxonomiccompletion
aresupportedbyasynonymcacheandaproject-leveltaxon-
omy cache.
Let𝐼𝑟denoteareferenceimageand𝑇𝑟itscompletedtax-
onomic prompt. Their normalized BioCLIP representations
are
̂𝐯𝑟=𝑓img(𝐼𝑟)
‖𝑓img(𝐼𝑟)‖2,̂𝐭𝑟=𝑓text(𝑇𝑟)
‖𝑓text(𝑇𝑟)‖2.(6)
Visualandtextualembeddingsarestoredasseparaterecords
in the same vector collection, preserving modality-specific
information for independent evaluation during retrieval.
Given a candidate crop𝐼𝑞, its normalized BioCLIP vi-
sual embedding is
̂𝐯𝑞=𝑓img(𝐼𝑞)
‖‖‖𝑓img(𝐼𝑞)‖‖‖2.(7)
Taxonomic prediction is performed using two complemen-
tary classifiers operating on the frozen BioCLIP embed-
dings.
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 11 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Forataxon𝑐representedby𝑁𝑐referenceimageembed-
dings, the centroid is computed as
𝝁𝑐=∑𝑁𝑐
𝑖=1̂𝐯𝑐,𝑖
‖‖‖∑𝑁𝑐
𝑖=1̂𝐯𝑐,𝑖‖‖‖2,(8)
wherê𝐯𝑐,𝑖isthenormalizedembeddingofthe𝑖-threference
image for taxon𝑐. The query crop is compared with each
cached centroid using cosine similarity:
𝑠centroid(𝑞,𝑐)=̂𝐯⊤
𝑞𝝁𝑐.(9)
Averagingovermultiplereferenceobservationsreducessen-
sitivity to atypical viewpoints, noise, and annotation errors.
Centroids are constructed at class, order, family, genus, and
specieslevels,enablinghierarchicalfallbackwhenevidence
is insufficient for a reliable species-level decision.
ThesecondstrategyusesmulticlasslinearSupportVec-
torMachinestrainedoncachedvisualembeddings.Foreach
taxonomic level𝑙, the score vector
𝐳(𝑙)=𝐖(𝑙)̂𝐯𝑞+𝐛(𝑙),(10)
is converted to normalized confidence via softmax:
𝑝(𝑙)
𝑗=exp(
𝑧(𝑙)
𝑗)
∑
𝑘exp(
𝑧(𝑙)
𝑘).(11)
Centroid and SVM classifiers provide complementary
evidence: centroids preserve the geometric structure of the
BioCLIP embedding space and are straightforward to up-
date, while linear SVMs learn explicit decision boundaries
that improve discrimination between visually similar taxa.
Both operate on frozen embeddings without retraining the
multimodal model.
The final identification decision combines both clas-
sifiers. Predictions below a configurable threshold are re-
jected;whenspecies-levelevidenceisinsufficientbuthigher-
rank candidates agree, a broader taxonomic label (genus,
family) is retained instead of forcing an unreliable species
assignment. When multiple crops from the same frame
receive identical predicted labels, only the top-𝑁by score
are preserved to reduce visual redundancy before down-
streamindexing.Eachvalidatedcropisstoredwithmetadata
(specie, genus, family, order, class, score, source file and
acquisition timestamp).
Theidentificationlayerthusconvertsvisualembeddings
into traceable taxonomic hypotheses. Classification results
aresubsequentlyusedbytheretrieval-augmentedreasoning
layer to gather supporting biological, visual, temporal, and
operational evidence from the local knowledge base.
Finally, the species identification layer can also be ac-
cessed via the Web user interface (UI). In Mission Mode,
users can directly upload an image of marine fauna for
classification. The interface then returns predictions from
boththeSupportVectorMachine(SVM)andcentroid-based
models across five taxonomic levels. Figure 5 illustratesthe species identification interface. For brevity, the figure
displays predictions at the class level only; however, the
full output provides predictions for all five taxonomic lev-
els. This functionality provides researchers with immediate
taxonomic hypotheses while maintaining a simple, user-
friendly workflow.
4.3.5. Retrieval-Augmented Reasoning (RAG)
Once taxonomic candidates, event labels, or mission
observationshavebeenproduced,theretrievallayercollects
the evidence required by agents or human users. In both
cases, retrieval is performed before language-model infer-
ence so that the generated response remains grounded in
locally indexed mission and biological data.
The retrieval engine operates over the unified Chro-
maDB collection described in Subsection 4.3.3. Because
the collection contains both visual and textual records, the
system supports four complementary retrieval modalities:
1. text-to-text retrieval, in which a textual query is com-
pared with indexed document chunks, annotations,
and taxonomic prompts;
2. text-to-image retrieval, in which a textual query re-
trieves semantically compatible reference images or
mission observations;
3. image-to-image retrieval, in which a query crop is
compared with indexed visual reference embeddings;
4. image-to-textretrieval,inwhichaquerycropretrieves
taxonomicprompts,speciesannotations,orothertex-
tual descriptions represented in the shared BioCLIP
space.
For a textual researcher query𝑇𝑞, the corresponding
embedding is compared independently with the textual and
visual partitions:
𝑡→𝑡=TopK(𝑓text(𝑇𝑞),text),(12)
𝑡→𝑖=TopK(𝑓text(𝑇𝑞),image),(13)
wheretextandimagedenote the textual and visual parti-
tionsoftheunifiedcollection.Similarly,whenaqueryimage
orcandidatecrop𝐼𝑞isavailable,thesystemperformsimage-
to-image and image-to-text retrieval:
𝑖→𝑖=TopK(𝑓img(𝐼𝑞),image),(14)
𝑖→𝑡=TopK(𝑓img(𝐼𝑞),text).(15)
Thecompatibilityofthesefourretrievalmodesresultsfrom
the shared multimodal embedding space learned by the
CLIP-based encoder.
To prevent one modality from dominating the final con-
text,thedeployedretrievalconfigurationretainsacontrolled
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 12 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
mixture of six visual records and four textual records for
a standard top-10retrieval operation. The visual subset
provides direct evidence from similar organisms, reference
images, and mission frames, whereas the textual subset in-
troduces explicit taxonomic annotations, scientific descrip-
tions,sensor-logsegments,andotherstructuredinformation.
This6∶4compositioncontrolsthetypeofevidenceprovided
to the reasoning agents.
The retrieval passes are executed separately and their
results are interleaved before final selection. This prevents
the substantially larger visual collection from occupying all
availableretrievalpositionsandensuresthattextualevidence
remains represented. When a modality filter is explicitly
requested, retrieval can instead be restricted to either image
or text records.
Each similarity-search pass initially retrieves more can-
didates than the final requested number. The candidates are
ordered according to vector distance and then filtered to
reduce redundancy. In particular, retrieved frames originat-
ing from the same video are subjected to temporal dedu-
plication: frames occurring within a three-second interval
of an already retained frame are discarded. This procedure
reducestherepeatedretrievalofnearlyidenticalconsecutive
observations while preserving temporally distinct events.
Every retained result is represented as a LangChain
Documentcontainingitsindexedcontentandassociatedmeta-
data. The retrieval engine augments this metadata with the
vector distance and the origin of the search pass. Results
obtained from visual and textual queries are subsequently
merged, duplicate identifiers are removed, and the final list
is truncated to the configured retrieval size.
The metadata available to the reasoning layer depend
on the retrieved modality and source dataset. Taxonomic
referencerecordsmayprovidespecies,genus,family,order,
andclass,togetherwiththesourceimagepathandfilename.
Mission video records may provide the source video, crop
orframepath,acquisitiontimestamp,detectionlabel,detec-
tionconfidence,BioCLIP-2predictionandsimilarityscores.
Sensor-log and tabular records may include the source file,
file name, and the start and end timestamps of the retrieved
interval.
BeforebeingpassedtotheQueryAgentorReportAgent,
the raw retrieval results are transformed into a structured
context package:
={𝐶text,,image,𝑆retrieval},(16)
where𝐶textis the assembled textual context,is the set
of loaded images,imagecontains their source paths, and
𝑆retrievalis a compact summary of the retrieved evidence.
Textual records are separated according to their origin.
Sensor-log and CSV chunks are formatted with their source
file and temporal interval:
[Source: file | time: start→end]
whereas biological annotation records are formatted using
the available taxonomic hierarchy. For example, a fish an-
notation may contain the species, genus, family, order, andclass followed by its descriptive text. Visual records also
contributetextualmetadatatothecontext.Anidentifiedref-
erenceimagecontributesitstaxonomicannotation,whereas
anunidentifiedmissionframecontributesitssourcevideoor
image file and acquisition timestamp.
Thetextualcontextislimitedtoaconfigurablemaximum
length,setto4000charactersinthecurrentimplementation.
When grouped video summaries are available, they are
inserted before the remaining retrieved chunks so that the
principal temporal and ecological evidence is preserved if
truncation becomes necessary.
A maximum of four retrieved images is loaded into the
multimodal context package. The images are accessed only
after retrieval, resized to1344×756pixels, converted to
RGB, JPEG-encoded, and represented as base64 strings.
This just-in-time image loading avoids placing complete
image contents in the vector database while allowing the
downstreamreasoningmodeltoinspectthemostrelevantvi-
sualevidence.Theoriginalfilepathsareretainedseparately
for logging and traceability.
Forvideo-orientedqueries,thesystemadditionallysup-
ports grouped crop retrieval. In this mode, image records
belonging to a selected source video are first grouped using
reliable taxonomic or detection metadata. A species label is
considered when an authoritative species value is available,
when the BioCLIP score exceeds the configured threshold,
or when a sufficiently confident non-generic detection label
is present. Records without reliable labels are grouped ac-
cording to the similarity of their normalized visual embed-
dings.
Let𝐠𝑗denote the normalized centroid of an existing
videogroupand𝐯𝑖theembeddingofanunlabelledcandidate
crop. The candidate is assigned to the most similar group
when
max
𝑗(𝐯⊤
𝑖𝐠𝑗)≥𝜏𝑔,(17)
where𝜏𝑔istheconfiguredgroupingthreshold.Otherwise,a
newvisualgroupiscreated.Inthecurrentconfiguration,the
default similarity threshold is0.86.
For each group, the system selects one representative
crop using the strongest available metadata confidence and
querysimilarity.Thecorrespondinggroupedsummarycon-
tains the number of detections, first and last observation
times,temporalsegments,representativefile,representative
timestamp, best and mean confidence or similarity, and the
available indexed-label evidence. When trained centroid or
linear SVM models are available, the summary may also
include the three strongest classification candidates at the
class, order, family, genus, and species levels.
The grouped retrieval mode therefore prevents a long
video sequence containing repeated observations of the
same organism from overwhelming the reasoning context.
Insteadofreturningeverysimilarcrop,itprovidesonevisual
representative per group together with a structured account
of its occurrence frequency and temporal distribution.
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 13 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Thecontextpackagerfinallyproducesaconciseretrieval
summary indicating the amount and type of evidence ob-
tained, such as the number of sensor-log chunks, textual
fishannotations,visualfishannotations,loadedimages,and
grouped video representatives. This summary allows the
reasoning agents to determine whether sufficient evidence
wasretrievedbeforeproducinganecologicalinterpretation.
The resulting structured package is supplied to the lo-
cal instruct model, Qwen3.5-4B, through the Ollama infer-
ence runner. The system employs this package to generate
grounded answers for researchers and to produce detailed
species reports, mission summaries, temporal activity de-
scriptions, and anomaly explanations. Rather than relying
exclusively on the parametric knowledge of the underlying
language model, the framework cites retrieved source files,
timestamps, taxonomic metadata, and visual observations
directly.
In high-activity Mission Mode, the instruct model can
remain resident in GPU memory to reduce repeated model-
loading and generation latency. In autonomous or energy-
constrained operating states, it can be unloaded after query
execution to reduce steady-state memory use and power
consumption.
Retrieval-augmented reasoning does not replace the
species identification layer. The identification layer gener-
ates taxonomic hypotheses from the visual observations,
whereas the retrieval layer gathers the biological, visual,
temporal, and operational evidence associated with those
hypotheses.Thelanguagemodelsubsequentlyusesthisevi-
dencetoexplainandcontextualizetheobservationswithout
modifying the underlying taxonomic predictions.
4.4. Multi-Agent Control Framework
Thehigh-levelbehaviorofthemonitoringstationiscon-
trolled by a local multi-agent framework. This framework
coordinates retrieval, structured analysis, energy manage-
ment, reporting, hardware configuration, and autonomous
post-processing. Instead of relying only on fixed rules,
the system assigns specialized responsibilities to different
agents. This makes the station more adaptable to changing
mission objectives, environmental conditions, and power
constraints.
4.4.1. Router Agent
TheRouterAgentisresponsiblefororchestratingqueries
in a heterogeneous, multimodal search environment. Rather
than relying on static mapping, it dynamically determines
the target database collections, the search modalities (tex-
tual, visual, or hybrid), and the downstream processing
pathway.
To achieve this, the agent is structured as an interac-
tive reasoning loop following the THINK–ACT–OBSERVE
paradigm. Before finalizing a retrieval route, the agent can
actively probe candidate collections. Through a probing (or
“peeking”)mechanism,itretrievesandinspectssmallmeta-
datasamplesfromspecificcollections,usingtheseempirical
observations to resolve query ambiguity.To assist the agent with complex taxonomic classifi-
cation queries, the system incorporates a hybrid context-
enrichmentstepbeforethereasoningloopbegins.Ifaquery
contains taxonomic keywords or is accompanied by an im-
age, the system runs a fast, non-parametric classification
step using pre-calculated reference centroids. The top taxo-
nomicpredictionsfromthisstepareinjecteddirectlyintothe
agent’s reasoning context as high-confidence prior beliefs,
guiding it toward the correct reference database.
For queries requesting visual species evidence (e.g.,
listing visually different species in a specific video), the
agent enforces a crop-level grouping guardrail. This path
bypassesnaiveglobalsimilarityretrievalandinsteadinvokes
a specialized visual-clustering pipeline that groups similar
objectdetections.Alternatively,forquantitativeortemporal
queries(e.g.,speciesoccurrencetimelinesorvideodatabase
inventories), the agent can activate an analytical fast path.
This path bypasses the generative language model entirely,
executing database-level metadata aggregations and report-
ing the results directly.
Finally, to handle edge cases or allow direct developer
intervention,thearchitecturesupportsadeterministictarget
override. When an explicit collection target is provided by
the user, the routing agent’s reasoning loop is bypassed en-
tirely,andqueriesarerouteddirectlytothemappedreference
or multimodal collections.
4.4.2. Analyst Agent
TheAnalystAgentisdesignedtoresolvequeriesthatare
better addressed via structured computation than through
heuristic generation. It compiles crop-level metadata (such
as species predictions, BioCLIP-2 confidence scores, video
names, and frame timestamps) extracted from the vector
database into a local Pandas DataFrame. An auxiliary text-
only model then generates targeted Python code to perform
statistical tasks, including species occurrence counting,
temporal distribution timelines, co-occurrence intervals,
and detection-frequency estimations. We evaluated both
Qwen3.5-4BandOrnith-9B;thelatterdemonstratedsignifi-
cantlysuperiorperformance,exceedinginitialexpectations.
To guarantee system safety and execution determinism,
the generated code is executed within a restricted local
namespace that isolates the system. File-system operations
and hazardous python built-ins are dynamically blocked,
and the code is statically validated using an abstract syntax
tree (AST) parser prior to execution. If a runtime error
is encountered, a bounded self-correction feedback loop is
initiated to repair the code. The final outputs, including
timestamps normalized toHH:MM:SSformat, are printed and
returned as structured Markdown tables or lists.
Becausethiscomputationalpathwaybypassesthelarger
vision-languageinstructmodelforpurelyquantitativequeries,
it reduces system latency and power consumption.
4.4.3. Energy and Sensor Management Agent
The Energy and Sensor Management Agent is respon-
sible for dynamically optimizing the power state of the
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 14 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Figure 9:Unified agentic reasoning and reporting layer. Incoming researcher queries or automated system triggers are evaluated
by a RouterAgent, which selects the appropriate retrieval, analysis, or reporting pathway. The agents interact with local memory,
the local instruct model, and hardware-management modules to generate grounded outputs for researchers.
stationtoensureitsurvivesthedesignatedmissionduration.
Ratherthanrelyingonstaticpowerconfigurations,theagent
continuouslycomputesareal-timepowerbudgetbycompar-
ing remaining battery capacity (Wh) against the remaining
mission time (hours).
To evaluate current consumption, the agent monitors
real-time system metrics—including CPU, GPU, and aux-
iliary power rails via Tegrastats on the primary Jetson
compute module—alongside a constant baseline approxi-
mation of the always-on sentinel microcontrollers. Based
on these variables and the recent event detection rate, the
agent dynamically adjusts software- controlled parameters
toalignthepowerdrawwiththeremainingenergyreserves.
These adjustments include tuning camera capture frame
rates(FPS),lengtheningorshorteningautonomouscompute
cycle intervals, and modifying the keep-alive duration of
cached model weights.
Furthermore, the agent determines the sentinel confi-
dence thresholds running on the microcontrollers to control
hybrid wake-up triggers, deciding when to transition the
primary compute module from a powered-off state directly
into high-fidelity processing. In this manner, energy man-
agement is treated as an active, context-aware control loop
that balances taxonomic accuracy with long-term hardware
survivability on batteries.
4.4.4. Dynamic Model Deployment Agent
The Dynamic Model Deployment Agent manages the
dynamic adaptation of the low-power sentinel tier. Ultra-
low-power microcontrollers offer extreme energy efficiency
but are constrained by strict on-chip memory limits, pre-
ventingtheconcurrentdeploymentofmulti-taskneuralnet-
works. The agent resolves this limitation by dynamically
selecting and flashing specialized, highly-quantized neural
networkweightsontothemicrocontrollersbasedondeploy-
ment locations, real-time battery status, seasonal predic-
tions, and researcher-defined mission priorities.Forinstance,theagentcanreconfigurethesentinellayer
to prioritize cetacean acoustic monitoring in one zone, and
hot-swap to visual invasive species detection in another. To
prevent flashing failures, the agent queries the model col-
lection, verifying candidate model sizes, hardware register
constraints, compatibility matrices, and past flashing histo-
riesbeforecommittingtoahardwareupdate.Thisorchestra-
tion enables a highly specialized sentinel layer to maintain
runtime flexibility across diverse long-term missions.
4.4.5. Reporting Agent
The Reporting Agent converts the station’s indexed
videologsintoresearcher-facingsituationsummaries.Rather
thanpresentingaflatlistofeventdetections,theagentaggre-
gates raw vector database metadata to produce a structured,
high-level overview of the survey’s progress and findings.
To generate a report, the agent compiles a determin-
istic database snapshot of the video assets. This snapshot
includes global metadata coverage statistics, the number of
processed video files, overall taxonomic species classifica-
tion distributions, and chronologically segmented temporal
detectionevents(groupingconsecutiveoccurrencesofindi-
vidual species within a configurable temporal window). A
local language model then synthesizes this raw structured
data into a cohesive, scientist-oriented narrative.
The resulting Markdown (.md) report is saved to a des-
ignated local directory with a query-specific, timestamped
filename. This workflow enables the low-bandwidth station
tostoreandtransmitcompact,highlydescriptivesummaries
rather than raw, high-resolution sensor streams when satel-
lite or acoustic telemetry connections become available.
5. Low-Power Sentinel Subsystems
Thissectiondetailsthetwospecializedlow-consumption
subsystemsdeployedaroundtheJetsonmasternode:thevi-
sualdetectionsubsystemontheMAX78002andtheacoustic
detection subsystem on the MAX78000.
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 15 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Figure 10:Visual detection subsystem.
5.1. Visual Detection Subsystem
Theproposedvisualsubsystemimplementsasequential,
shared-busdataarchitecturetooptimizememoryutilization
duringhardware-acceleratedexecution.Dataacquisitionbe-
gins through both a MIPI CSI-2 and DVP interface, which
captures RGB565 image frames and streams them sequen-
tially into external QSPI SRAM using a dedicated line
handler. The system was tested with two cameras, OV5640
and OVM7692.
During preprocessing, the microcontroller unit (MCU)
retrieves the stored frame in distinct chunks, transcodes the
pixeldatainto24-bitRGB888format,packstheresultsinto
32-bitwords,andloadsthemdirectlyintotheconvolutional
neuralnetwork(CNN)hardwareFIFO.Tominimizeoverall
energy consumption, the MCU transitions into a low-power
sleep state during inference and delegates computation of
bounding box regressions and class logits to the hardware
accelerator. Upon inference completion, the MCU resumes
execution to poll the output registers. It applies a softmax
activation to derive class probabilities and transforms the
spatial outputs into YOLO-style bounding box coordinates.
Redundantlocalizationsaresystematicallyeliminatedusing
a Non-Maximum Suppression (NMS) algorithm based on
Intersection over Union (IoU) filtering. Finally, the raw
image is retrieved row by row from SRAM and transmitted
over SPI to a TFT display with overlaid bounding boxes
and confidence scores. While real-time display rendering
is primary for validation, operational deployments bypass
the display, opting instead to write the image to an SD
card or issue a GPIO interrupt to wake the Jetson hostmodule. Figure 10 represents the sequence diagram of this
subsystem.
Two lightweight vision models are tested: FPN De-
tector and TinyissimoYOLO. Their architectures are de-
scribed in Sections 5.1.1 and 5.1.2. A major constraint on
theMAX78002isthatthehardwaredictateshowthemodel
should be structured. The architecture supports up to 128
layers, 2048 input and output channels per layer, and image
dimensions up to 2047 pixels. Supported operations in-
clude Conv1D, Conv2D, ConvTranspose2D, pooling, fully
connected layers, and element-wise arithmetic operations.
Weightscanbequantizedto1,2,4,or8bits,andactivations
are processed in signed 8-bit format.
The device provides approximately 2.34 MiB of dedi-
cated weight memory and 1.28 MiB of data SRAM dis-
tributed across 16 memory banks. Convolutional layers
are restricted to specific kernel sizes and strides optimized
for hardware efficiency, with support for depthwise con-
volutions and streaming mode to process inputs exceed-
ing on-chip memory capacity. Convolutional layers on the
MAX78002arelimitedtothefollowinghardware-supported
configurations:
•Conv2D:kernel size1×1or3×3only.
•Conv2D stride:fixed to1×1.
•Conv2D padding:0,1, or2.
•Conv2D dilation:1to16.
•Groups:standard convolution (𝑔= 1) or depthwise
convolution (𝑔=𝐶in=𝐶out).
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 16 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
•Conv1D:kernel size1to9.
•Conv1D stride:fixed to1.
•Conv1D padding:0to2.
•ConvTranspose2D:kernel size3×3only.
•ConvTranspose2D stride:fixed to2×2.
•ConvTranspose2D output padding:fixed to1.
Consequently, convolutions such as5×5,7×7, stride-
2Conv2D, stride-4Conv2D, or arbitrary grouped convolu-
tions are not directly supported by the accelerator and must
be decomposed into supported operations during model
conversion.Fullyconnectedlayersareimplementedas1×1
convolutions and are constrained by input flattening lim-
its. The platform also uses hardware-accelerated activation
functions.
5.1.1. FPN Detector
The first tested model is a Feature Pyramid Network
(FPN) object detection architecture developed by the MAX
78002 team and tailored specifically for ultra-low-power,
real-time edge AI processing. Operating on an RGB in-
put image of size256×320, the network uses two initial
fused convolutional layers to project the input to a 64-
channel space before feeding it into a highly optimized
ResNetBackbone. This backbone uses strided max-pooling
layersdirectlyembeddedwithinitsresidualblockstodown-
samplethespatialdimensionsacrosssixdistinctsub-blocks,
producingfourmultiscalefeaturemapsatresolutionsmatch-
ing32×40,16×20,8×10, and4×5.
These representations are then aggregated by a top-
down FPN module that uses1×1lateral convolutions to
unify all pyramid feature planes to a 64-channel baseline,
together with3 × 3transposed convolutions for spatial
upsampling and element-wise additions to reduce aliasing
artifacts. The resulting pyramid feature maps,𝐏0to𝐏3,
are routed in parallel to dual, weight-sharing subnetworks
for classification and bounding box regression. These sub-
networks simultaneously evaluate 10,200 anchor boxes per
frame generated across variable aspect ratios(0.5,1.0,2.0)
and scale modifiers.
By using proprietary fused primitives the architecture
maximizeshardwarepipeliningefficiency.Thisallowscom-
plex structural semantics and anchor-based post-processing
operations,includingJaccard-coordinatedecodingandNon-
Maximum Suppression, to execute within the strict energy
and memory constraints of the MAX78002 microcontroller
platform. More details are provided in the repository and
documentation supplied by Analog Devices [4].
5.1.2. TinyissimoYOLO
TinyissimoYOLO is an ultra-lightweight object detec-
tion architecture engineered specifically for extreme edge-
computing environments and resource-constrained micro-
controllers. It was developed by Julian Moosmann at the
Center for Project Based Learning, ETH Zurich [20]. Thenetwork relies on a streamlined, strictly sequential convo-
lutionalneuralnetworkbackboneconsistingofstandard2D
convolutionallayers,typicallyusing3×3kernelswithReLU
activations, interspersed with2×2max-pooling layers.
Thistopologyprogressivelydownsamplesthespatialdi-
mensionsoftheinputimagewhileexpandingchanneldepth,
generally scaling from 16 up to 128 channels. To strictly
minimize computational overhead and memory footprint,
thearchitectureavoidscomplexmechanismssuchasfeature
pyramids. Instead, the final pooled feature map is routed
directly into a compact detection head, often implemented
asadenselayerora1×1convolution,whichoutputsastruc-
tured tensor containing bounding box coordinates and class
probabilities. This design prioritizes low parameter count
anddeterministicexecution,enablingreal-timeinferenceon
theMAX78002microcontroller.Additionalimplementation
details and model variants are provided in the associated
repository [20].
5.2. Acoustic Detection Subsystem
The developed acoustic classification system imple-
ments a direct-memory-access-driven continuous process-
ing pipeline to detect marine mammal vocalizations on a
resource-constrained hardware architecture. Audio is sam-
pled at 16 kHz through an I2S interface and buffered
dynamically. Incoming data is digitally high-pass filtered
to eliminate DC offset and evaluated against an amplitude
threshold to suppress inference on ambient noise.
Forvalidacousticevents,asoftware-baseddigitalsignal
processing routine generates a log-Mel spectrogram by ap-
plying a periodic Hann window, executing a 512-point real
Fast Fourier Transform, and mapping the power spectrum
througha64-binsparsetriangularMelfilterbank.Theresult-
ing feature matrix is quantized to 8-bit integers and loaded
directlyintothestaticrandom-accessmemoryofadedicated
convolutional neural network accelerator.
The deployed classifier employs a compact five-stage
convolutional neural network optimized for execution on
the MAX78000 hardware accelerator. The network accepts
a single-channel 96×64 log-Mel spectrogram as input and
progressivelyextractshigher-levelacousticfeaturesthrough
successive3×3convolutionallayers,eachfollowedbybatch
normalization, ReLU activation, and 2×2 max-pooling (ex-
cept for the first layer). The feature maps are reduced from
16 to 64 channels while the spatial resolution is gradually
compressed, yielding a compact 64×3×2 representation. A
final 1×1 convolution preserves the feature depth before
flatteningtheoutputintoa384-elementfeaturevector,which
is processed by a fully connected layer to classify the input
into one of 13 categories, comprising 12 cetacean vocal-
ization classes and one background noise class. The archi-
tecture is specifically designed to minimize computational
complexity and memory usage while maintaining sufficient
representational capacity for accurate embedded acoustic
classification.
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 17 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Figure 11:Acoustic detection subsystem.
During CNN inference, the primary microcontroller
yields execution to the hardware accelerator to reduce dy-
namic power consumption. Once inference is complete,
the accelerator returns a vector of class scores, which is
processedusingasoftmaxfunctiontoobtaintheconfidence
associated with each category. If the highest-confidence
prediction corresponds to a cetacean vocalization rather
than the background noise class, the associated raw audio
bufferisautomaticallyarchivedtotheintegratedSDcardfor
subsequent empirical validation. The complete processing
sequence of the acoustic subsystem is illustrated in Figure
11.
6. Experimental Results and Evaluation
This section evaluates the operational efficiency, detec-
tion accuracy, and system-wide energy consumption of the
proposed hierarchical agentic underwater monitoring sys-
tem.Theevaluationisstructuredtofirstbenchmarktheultra-
low-power sentinel tier independently, and then to validate
the high-fidelity master tier and the overall power savings
achieved by the hardware-switched cascade architecture.
6.1. Experimental Setup and Benchmarking
Environment
To validate the system under realistic edge constraints,
the multi-agent framework and high-fidelity models wereexecuted on a Seeed reComputer J4012 powered by an
NVIDIA Jetson Orin NX module (16GB RAM) . The con-
tinuous sentinel tier consisted of a MAX78002 EV kit and
a MAX78000 FTHR board. The NVIDIA Jetson Orin NX
and Maxim Integrated MAX78000/MAX78002 were pow-
ered by an Aim-TTi CPX400DP DC power supply. Sys-
tem current, voltage, and total power consumption for the
MAX78000 FTHR board and the Seeed reComputer J4012
(Jetson Orin NX) were monitored and recorded separately
usinganexternal,inlineRohde&SchwarzHMC8015digital
poweranalyzer.Incontrast,theMAX78002EVKittracked
its power consumption internally using its onboard power
accumulator module.
The evaluation datasets were already introduced in sec-
tion 3.
6.2. Sentinel Evaluation
The Sentinel Tier was evaluated under the deployment
constraintstargetedbytheproposedlow-powerarchitecture.
The visual sentinel was assessed using mAP@50, while the
acousticsentinelwasevaluatedusingaccuracy,recallandF1
score. The visual sentinel was trained for 100 epochs using
the same train/validation/test split proposed by the original
datasetauthors.TrainingemployedtheAdamoptimizerwith
an initial learning rate of 0.001, a batch size of 64, and a
cosineannealinglearningrateschedule.Quantization-aware
training (QAT) was enabled from epoch 73 onward. As
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 18 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Table 4
Sentinel-tier core performance evaluation metrics.
Subsystem (Chip) Model Architecture Target Task Evaluation Metrics
Vision (MAX78002)FPN Detector Fish Detection mAP@50: 78.4%
TinyissimoYOLO Fish Detection mAP@50: 70.75%
Acoustic (MAX78000) Mel-Spectrogram CNN 12 Marine MammalsAccuracy: 98.99±0.11%
Recall: 95.97±0.86%
F1: 96.36±0.52%
described in Section 3.1, stochastic data augmentation was
applied throughout training, which was further restricted to
images containing at most five annotated bounding boxes
to better reflect the target deployment scenario. All visual
model training was performed on the IFREMER HPC clus-
ter.Fortheacousticsentinel,evaluationwasconductedusing
five-fold cross-validation. The model was trained for 50
epochs using the Adam optimizer (weight decay10−4), an
initial learning rate of 0.001, and a batch size of 128, with
QAT enabled from epoch 37 onward. Table 4 summarizes
the resulting performance of both Sentinel Tier models.
6.3. Experimental Validation of the Master Tier
The primary objective of the experimental evaluation
was to verify the feasibility of the proposed edge-based
multi-agent system under realistic deployment constraints.
While the fish identification component could be quantita-
tivelyassessedusingcuratedbenchmarkdatasets,evaluating
the complete agentic RAG pipeline proved considerably
morechallengingduetotheabsenceofstandardizeddatasets
andestablishedmetricstailoredtothisapplication.Further-
more,thescopeofthisworkdidnotpermitthedevelopment
ofdedicatedbenchmarksfortherouting,analyst,andreport-
ing agents. Consequently, the evaluation combines rigorous
quantitative experiments for the fish identification module
withqualitativevalidationoftheremainingagents,focusing
on their functional correctness, integration, and suitability
for low-power edge deployment.
The fish identification layer was evaluated using two
complementary protocols. First, a cross-dataset evaluation
wasconductedacrossthesixreferencecollectionsdescribed
inTable3:FishBase/SeaLifeBase,WildFish++,Fish-Vista,
FishNet,BioTrove restrictedto marinespecies, andOzFish.
In this setting, each dataset was used as a reference collec-
tion and evaluated against the others in order to quantify
cross-domain generalization. OzFish was used as the real-
condition reference dataset, making it the main target for
assessing deployment robustness. Since the retrieval and
classification pipeline is deterministic once the embeddings
are fixed, this cross-dataset evaluation was performed once.
Second,anintra-datasetevaluationwasconductedinde-
pendentlyforeachdataset.Forthisprotocol,10independent
species-stratifiedsplitsweregenerated,with80%ofthesam-
ples embedded into the vectorized reference space and the
remaining20%usedfortesting.Resultsarereportedasmean
±standard deviation. Classification was evaluated at five
taxonomic levels: class, order, family, genus, and species.
Two ranking-based metrics were used: Top-1 accuracy andTop-5accuracy.Thecross-datasetresultsaresummarizedin
Table 5, while the intra-dataset split results are reported in
Table 6.
For both cross-dataset and intra-dataset evaluations,
the tables present results for the SVM classifier, which
achievedsuperiorperformanceinmostcases.Thisoutcome
was expected, as centroid-based methods risk obscuring
intra-speciesfinedetails,particularlywhenmorphologyand
coloration vary across age or sex.
The remaining components of the system were evalu-
ated qualitatively through functional validation. Given the
specialized nature of the proposed architecture, no pub-
licly available benchmarks currently exist to objectively as-
sesstheperformanceofanedge-deployedmulti-agentRAG
system operating across heterogeneous scientific databases.
Consequently, the evaluation focused on verifying that the
agents performed their intended roles reliably while satis-
fying the power and computational constraints required for
autonomous edge deployment.
For the routing architecture, testing confirmed the cor-
rect selection of the appropriate knowledge source and suc-
cessful execution of the retrieval pipeline under realistic
operating conditions. Although large-scale benchmarking
involving numerous vector databases with overlapping se-
manticdomainsconstitutesanimportantdirectionforfuture
work, the present experiments demonstrate the practical
feasibility of the routing strategy within the target hardware
and energy constraints.
Testing of the AnalystAgent confirmed that structured
codegenerationandexecutioncanbeperformeddirectlyon
theJetsonedgeplatform.Ininitialevaluations,theQwen3.5-
4B model exhibited significant performance limitations on
complextasks:theReActexecutionloopfrequentlyrequired
up to eight iterations to reach a solution, primarily due to
constraints on available functions/libraries and an inability
toprocessthefullcontextsimultaneously.Incontrast,testing
with the Ornith 9B model exceeded expectations, generat-
ing executable Pandas code on the first iteration in most
cases,andrequiringatmosttwoloops.Furthermore,because
Ornith 9B is a text-only model, it eliminates the unneces-
sary computational overhead of vision-language modalities
present in Qwen3.5-4B—a key advantage for the text- and
data-centric requirements of the AnalystAgent. One opera-
tional requirement identified on the Jetson platform is the
necessity of clearing the NVIDIA GPU cache prior to ini-
tializingtheAnalystAgenttopreventCUDAout-of-memory
(cudaMalloc) errors. Overall, these results demonstrate that
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 19 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Table 5
Cross-datasettaxonomicidentificationmatrix.Rowsindicatethedatasetusedastheembeddedreferencecollection,whilecolumns
indicate the dataset used for testing. Each cell reports a compact taxonomic performance signature in the order Class, Order,
Family, Genus, and Species. Values are reported as Top-1/Top-5 accuracy percentages. OzFish is used as the real-condition
deployment reference.
Reference→Test FishBase/ WildFish++ Fish-Vista FishNet BioTrove OzFish
SeaLifeBase Marine Real
FishBase/SeaLifeBase–C: 98.7/100.0
O: 78.5/95.1
F: 75.4/90.4
G: 65.3/80.0
S: 51.9/67.8C: 99.3/100.0
O: 81.5/96.7
F: 75.4/91.1
G: 45.3/66.0
S: 11.2/23.9C: 98.0/99.9
O: 80.6/96.6
F: 73.3/88.8
G: 70.4/84.4
S: 60.0/70.6C: 97.7/99.8
O: 75.3/95.9
F: 60.6/83.6
G: 48.2/71.6
S: 24.7/43.1C: 97.4/100.0
O: 73.0/92.9
F: 13.4/42.9
G: 13.9/30.0
S: 3.2/6.8
WildFish++C: 88.5/89.7
O: 66.4/83.3
F: 46.3/57.5
G: 35.1/41.5
S: 23.5/24.7–C: 99.4/100.0
O: 70.3/94.5
F: 67.2/83.5
G: 21.7/32.6
S: 8.4/19.1C: 98.4/99.9
O: 77.9/94.6
F: 66.6/79.4
G: 52.3/61.4
S: 39.4/44.8C: 98.3/99.6
O: 71.6/93.0
F: 56.4/72.4
G: 37.1/49.3
S: 20.5/26.1C: 95.5/100.0
O: 70.1/93.4
F: 15.3/44.5
G: 11.5/27.7
S: 2.6/6.7
Fish-VistaC: 84.8/85.3
O: 53.3/71.7
F: 25.9/38.7
G: 12.0/18.5
S: 3.4/4.7C: 96.2/96.5
O: 62.2/83.6
F: 40.2/61.1
G: 19.7/28.3
S: 5.4/7.5–C: 92.5/93.7
O: 59.7/80.8
F: 36.3/52.9
G: 19.2/28.1
S: 9.2/14.6C: 93.0/93.6
O: 57.4/78.1
F: 34.8/55.3
G: 11.5/20.3
S: 1.9/3.6C: 98.6/98.6
O: 75.6/89.6
F: 16.2/52.5
G: 13.2/29.0
S: 0.7/2.1
FishNetC: 88.8/90.0
O: 72.2/87.5
F: 57.2/68.3
G: 64.2/70.5
S: 65.0/67.2C: 98.9/100.0
O: 77.1/96.2
F: 73.2/90.4
G: 64.3/81.0
S: 47.2/65.9C: 99.3/99.9
O: 79.0/96.1
F: 74.9/91.1
G: 38.5/62.4
S: 14.9/30.6–C: 98.4/99.9
O: 73.7/95.5
F: 63.0/82.0
G: 41.3/65.0
S: 17.1/33.6C: 95.5/100.0
O: 70.4/93.3
F: 14.1/33.9
G: 9.6/19.8
S: 1.2/2.8
BioTrove MarineC: 88.5/89.6
O: 68.1/85.8
F: 46.7/63.1
G: 35.9/52.7
S: 19.7/32.0C: 98.9/99.8
O: 77.2/95.5
F: 72.6/88.9
G: 58.8/74.4
S: 44.3/59.9C: 99.1/99.7
O: 74.6/94.5
F: 65.9/85.2
G: 34.3/58.1
S: 3.0/8.5C: 98.4/99.8
O: 79.0/96.2
F: 69.0/85.5
G: 58.0/75.9
S: 41.1/56.8–C: 96.2/100.0
O: 73.5/94.1
F: 16.7/47.5
G: 12.6/33.4
S: 4.2/8.2
OzFish Real– – – – – –
Each non-diagonal cell reports five rows: C, O, F, G, and S for Class, Order, Family, Genus, and Species. Each row follows the format
Top-1/Top-5 accuracy.
Table 6
Intra-dataset taxonomic identification evaluation across validated marine reference datasets. Results are reported as mean±
standard deviation over 10 species-stratified 80/20 splits.
DatasetClass Order Family Genus Species
T1 T5 T1 T5 T1 T5 T1 T5 T1 T5
Fish/SeaLife Base 96.8±0.1 99.6±0.0 79.4±0.2 96.9±0.1 70.4±0.2 88.6±0.2 63.1±0.2 83.4±0.1 43.0±0.4 61.3±0.3
WildFish++ 99.6±0.0 100.0±0.0 86.3±0.2 99.4±0.1 90.3±0.1 98.8±0.0 88.7±0.2 97.7±0.1 79.4±0.2 92.5±0.1
Fish-Vista 99.7±0.0 100.0±0.0 95.3±0.2 99.6±0.1 95.4±0.1 98.9±0.1 83.9±0.3 93.2±0.1 68.9±0.3 85.7±0.2
FishNet 98.9±0.0 100.0±0.0 86.3±0.1 98.5±0.0 78.9±0.1 90.6±0.1 75.8±0.2 89.4±0.1 60.2±0.2 73.3±0.1
BioTrove Marine 98.7±0.0 100.0±0.0 80.8±0.2 98.2±0.1 69.6±1.1 86.0±0.7 73.8±0.4 91.1±0.2 68.9±0.3 85.4±0.1
T1 and T5 denote Top-1 and Top-5 accuracy, respectively. Each value is reported as mean±standard deviation across 10 independent
species-stratified splits.
deploying capable 8B–9B parameter models on resource-
constrained edge hardware achieves both high execution
accuracy and system feasibility.
TheReportingAgentwasnotevaluatedusingadedicated
quantitative metric. Its behavior is largely deterministic,
producing structured Markdown reports from the validated
outputs of the upstream agents and the current station state.
Consequently, its assessment focused on verifying the cor-
rectness, consistency, and completeness of the generated
reports rather than measuring predictive performance.6.3.1. Energy Consumption Evaluation
Continuoussentinel-layerconsumption.Thevisualand
acoustic sentinels remain active continuously, establishing
thebaselineenergyfootprintofthesystem.Thiscontinuous
sentinel layer operates independently of the downstream
processing cycles, and its energy contribution is measured
in isolation to define the static power draw required for
uninterrupted environmental monitoring. Table 7 presents
the experimental evaluation of the visual sentinel subsys-
tem. We report both performances of Tinyissimo YOLO
and FPN Detector. The evaluation considers eight distinct
configurations: for both tested models employing either an
OV5640 (a 5 MP CSI camera) or an OVM7692 (a VGA
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 20 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Table 7
Experimental characterization of the continuously active visual sentinel subsystem (MAX78002). Note: Values reflect only the
deployable portion of the system, measured via the MAX78002 EVKIT internal power monitoring circuitry, excluding the rest of
the development board components.
Model Camera & ModeEnergy Per Inference
(mJ/inf)Active Power
mWIdle Power
mWLatency
msFPS
ms
TinyissimoYOLOOVM7692 (Inference Only) 9.2 90.65 11.3 116.4 8.6
OVM7692 (Capture + Inf) 11.97 86.57 11.3 159 6.3
OV5640 (Inference Only) 17.97 119 13.2 170 5.88
OV5640 (Capture + Inf) 23.65 103 13.57 264.3 3.8
FPN DetectorOVM7692 (Inference Only) 63.9 482.85 11.5 135 7.4
OVM7692 (Capture + Inf) 67.13 372.56 11.31 185.8 5.4
OV5640 (Inference Only) 63.5 480 13.2 135 7.4
OV5640 (Capture + Inf) 68.37 336.8 13.21 211.3 4.7
Table 8
Experimental characterization of the continuously active audio sentinel subsystem (MAX78000FTHR). Note: Due to hardware
monitoring limitations, values represent total board consumption, as isolated measurement of audio acquisition, CNN, and DSP
components only is not supported.
Audio ModeAverage Power
(mW)
Internal Microphone50
External Audio (FTHR CODEC)180
DVP camera), with measurements taken for both inference-
onlyandcapture-plus-inferenceoperationalmodes.Notably,
theOVM7692yieldssignificantlylowerpowerconsumption
than the OV5640. This improvement is primarily attributed
to its native hardware optimization for the MAX78002 EV
Kit—with which it is bundled as the reference sensor—as
well as highly tunable operational parameters, particularly
regarding clock speed control. Furthermore, the OVM7692
benefits from reduced sensor startup latency, lower initial-
ization overhead, and a simpler interface clocking scheme
inherenttoitslow-resolutionDVParchitecturecomparedto
the higher-resolution MIPI-CSI setup.
Measured processing blocks.In contrast to the contin-
uous sentinel layer, the Jetson module’s energy consump-
tion is highly dynamic and depends strictly on the deploy-
mentconfiguration.Thesystemutilizesthreedistinctoperat-
ingmodes:periodicduty-cycling,event-triggeredactivation
uponsentineldetection,andsustainedcontinuousoperation
featuring a graphical user interface.
Toevaluatetheenergyrequirementsacrossthesevarying
modes, the Jetson processing pipeline is decomposed into
ninediscrete,experimentallymeasurableoperationalblocks.
These blocks, characterized in Table 9, correspond to the
sequential tasks executed during system operation:
1. boot and initialization;
2. sentinel-data transfer;
3. video-frame extraction;
4. CLIP & Grounding Dino loading;
5. image/frame indexing;
6. CLIP similarity retrieval;7. Router–RAG request processing;
8. scientific report generation;
9. controlled shutdown.
Mathematical energy formulation.To extrapolate the
total daily energy consumption (𝐸total) across the three
deployment configurations, the system’s energy footprint is
modeled as the sum of the static sentinel baseline (𝐸sentinel)
and the dynamic Jetson processing load (𝐸dyn):
𝐸total=𝐸sentinel+𝐸dyn (18)
The continuous sentinel baseline is deterministic and
independent of the operating mode. It is defined as:
𝐸sentinel=𝑃sentinel ⋅𝑇day (19)
where𝑃sentinelis the combined average power of the active
sentinels(Tables8and 7)and𝑇dayisthe24-houroperational
period.
The dynamic energy,𝐸dyn, represents the variable con-
sumption of the Jetson module. Let𝐸𝑏𝑖denote the energy
requiredtoexecutethe𝑖-thprocessingblock(Table9),and𝑛𝑖
denoteitsdailyexecutionfrequency.Becausetheindividual
energy blocks already incorporate the idle power draw and
the time between operations is negligible, the dynamic load
is formulated solely as the sum of these discrete processing
events:
𝐸dyn=9∑
𝑖=1𝑛𝑖⋅𝐸𝑏𝑖(20)
The execution frequency𝑛𝑖strictly depends on the se-
lected operating mode:
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 21 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
Table 9
Experimental energy characterization of the discrete processing blocks utilized across the three deployment modes. Values are
reported for the exact workload specified during each measurement.
Processing block Measured workload Energy (Wh) Duration (s) Average power (W)
Boot and initialization One activation0.0434 16.374 9.55
Sentinel-data transfer1 MB(32 audio files of 32KB)0.26 136 6.88
Video-frame extraction1frame720∗360 0.0001 0.038 9.77
CLIP & Grounding Dino loading One loading sequence0.076 32 8.55
Image/frame indexing1frame720∗360 4.17×10−50.0137 10.96
CLIP similarity retrieval One query over49150indexed items0.046 20 8.28
Router–RAG request One inference request0.431 139 11.16
Scientific report generation One report from 187 indexed detections 0.646 177 13.14
Controlled shutdown One shutdown sequence0.014 8 6.3
•Periodic duty-cycling:The Jetson powers on at a
fixed time intervalΔ𝑡. The execution frequency for
core retrieval and indexing blocks is deterministic,
defined as𝑛𝑖=⌊𝑇day∕Δ𝑡⌋. The system undergoes a
controlled shutdown (Block 9) after each cycle.
•Event-triggered:The processing sequence is initi-
atedexclusivelybysentineldetectionevents.Thefre-
quency𝑛𝑖becomes a stochastic variable correspond-
ing to the expected daily event rate,𝜆event.
•Sustained continuous:The Jetson module remains
poweredindefinitelytosupportthegraphicalinterface
and real-time processing. The execution frequency
𝑛𝑖is driven by the volume of spontaneous scientist
requests (Block 7).
By defining hypothetical operational scenarios to estab-
lishthefrequencyvariables(𝑛𝑖),thisformulation,combined
with the measured energy blocks, provides a direct method
toestimatethetotaldailyenergyconsumptionofthesystem.
This consumption profile is subject to further optimization.
StructuralefficiencycanbeimprovedbyintegratingYOLO-
World, as discussed in Section 4.3.2, or by increasing the
serial communication baud rate from 115200 to a higher
operationalrate,therebyminimizingtheactivetransmission
time between components.
7. Discussion
The proposed system demonstrates that the integra-
tion of ultra-low-power sentinel sensing with high-fidelity
local multimodal reasoning can significantly enhance au-
tonomous underwater monitoring capabilities in network-
isolated/limited environments. By decoupling continuous
environmental surveillance from intensive data processing,
the architecture achieves a robust balance between energy
efficiency and scientific insight generation. This section
discusses the implications of our findings, highlighting the
advantages of the local Retrieval-Augmented Generation
(RAG) pipeline, while acknowledging operational trade-
offs such as inference latency. We also explore how this
framework can be extended to diverse marine applications
beyond standard biodiversity surveys.7.1. Advantages of Local Multimodal RAG and
Adaptive Identification
A key strength of the proposed architecture is its ability
toperformcomplextaxonomicidentificationwithoutrelying
oncloudconnectivityorpre-trainedclosed-setclassifiersfor
every species. The use of BioCLIP/OpenCLIP embeddings
withinalocalChromaDBvectordatabaseenablesaflexible,
retrieval-basedapproachtospeciesrecognition.Thismethod
allows researchers to dynamically update reference collec-
tionsbasedonspecificmissionobjectives(e.g.,focusingon
invasive species in one zone and endemic fauna in another)
without retraining the underlying neural networks.
The adaptive nature of this identification layer is partic-
ularly valuable for exploratory missions where target taxa
mayvaryorremainunknownuntildeployment.Bycombin-
ing centroid-based similarity search with linear SVMs, the
system provides hierarchical taxonomic hypotheses (from
classtospecieslevel)accompaniedbyconfidencescoresand
supporting visual evidence. This transparency allows scien-
tists to verify identifications directly at the edge, reducing
reliance on post-deployment manual annotation. Further-
more,theintegrationoftextualmetadata—suchasscientific
descriptions and habitat information—into the embedding
spaceenablessemanticqueriesthatgobeyondsimpleimage
matching, fostering a deeper understanding of ecological
contexts through local RAG pipelines. Another possibility
is using the localization and identification layers to count
fish: by cropping and then identifying the species, it will be
possible to keep a record and store the count as metadata.
7.2. Agentic Reasoning: Balancing Intelligence
and Latency
The multi-agent framework (RouterAgent, AnalystA-
gent, Reporting Agent) introduces a layer of autonomous
decision-making that transforms raw sensor data into struc-
tured scientific knowledge. While the execution of large
language models (LLMs) on edge hardware like the Jetson
Orin NX inherently involves higher latency compared to
simpleinferencetasks,thisoverheadisjustifiedbythevalue-
added services provided: natural-language query handling,
automatedreportgeneration,anddynamicresourcemanage-
ment.
The RouterAgent’s ability to intelligently route queries
between visual retrieval, textual analysis, and structured
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 22 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
computation ensures that computational resources are al-
located efficiently. For instance, deterministic statistical re-
quests are handled via the AnalystAgent using sandboxed
Python execution, bypassing the LLM when unnecessary.
Although real-time interaction may experience delays due
to model loading and generation times, this latency is ac-
ceptable for non-critical scientific inspection tasks (Mis-
sion Mode) where depth of analysis takes precedence over
millisecond-levelresponsiveness.Thesystem’sdesignprior-
itizescomprehensivesituationalawarenessandautonomous
reporting capabilities, which are crucial for long-term de-
ployments where human intervention is limited.
7.3. Energy Efficiency Through Hierarchical
Activation
The hierarchical master–satellite topology effectively
mitigates the high energy costs associated with continuous
GPUusage.BykeepingtheJetsonOrinNXpoweredoffex-
cept during scheduled cycles or event-triggered activations,
the system extends operational duration significantly com-
pared to always-on architectures. The MAX78000/78002
sentinels provide reliable first-stage filtering for visual and
acoustic events, ensuring that high-performance processing
is reserved for scientifically relevant occurrences.
Whilethesentinelmodelsoperateunderstricthardware
constraints, their optimized architectures achieve sufficient
accuracyfortrigger-basedactivationwithoutexcessivefalse
positives. The Energy Management Agent further enhances
sustainability by dynamically adjusting system parameters
based on remaining battery capacity and mission priorities.
Although current energy estimation relies on heuristic ap-
proximations, this proactive management strategy demon-
strates the feasibility of self-regulating power consumption
in remote underwater environments.
7.4. Limitations of the current framework
Even though the current system offers many benefits
in terms of power consumption, it is still limited by the
currentlyavailablehardwareandsoftware.Forinstance,the
MAX78000andMAX78002arehighlyefficientintermsof
powerconsumption,outperformingalmostanyavailablemi-
crocontrollerinenergyperinference.However,theypresent
severalissuesthatmayslowdownfuturedeployment.First,
thehardwareacceleratorislimitedexclusivelytoCNNmod-
els—and specifically, only those that respect the constraints
presented in Section 5, making it difficult to deploy models
that are already available online. Another problem is the
SDK; while most standard chips use classical deployment
through ONNX, the Analog Devices chips require training
and deployment through a specialized SDK, which slows
down the deployment process. On the other hand, currently
availableVLM(orclassicalLLM)modelsunder4Bparam-
etersarestillnothighlyreliable,especiallyforagentictasks.
Inthefuture,morespecializedsmallmodelsmayappear,and
thecurrentmodel-agnosticimplementationthroughOllama
will make upgrading and testing new models easier.7.5. Scalability and Concrete Underwater
Applications
Beyond standard fish detection and marine mammal
acoustic monitoring, the proposed architecture is highly
scalable to other critical underwater applications:
•InvasiveSpeciesMonitoring:Theadaptivereference
collection feature allows rapid deployment against
specificinvasivethreats(e.g.,LionfishorGreenCrab)
by simply uploading targeted image/text references.
The system can automatically alert researchers when
high-confidence detections occur, enabling timely
management interventions.
•Benthic Habitat Assessment:By integrating visual
similaritysearchwithtaxonomicmetadata,thesystem
canclassifyandquantifybenthiccommunities(corals,
sponges,seagrasses)overtime.TheRAGpipelinecan
generateautomatedhealthreportsbasedonchangesin
species composition or coverage area detected by the
sentinels.
•Anthropogenic Activity Tracking:The multimodal
natureofthesystemsupportsdetectionandclassifica-
tionofhuman-madeobjects(e.g.,fishinggear,debris,
orvessels).Acousticsignaturescombinedwithvisual
confirmationcanhelpmonitorillegalfishingactivities
or assess pollution levels in protected marine areas.
•ROV/AUV Integration:The lightweight, modular
design is suitable for integration into Remotely Op-
erated Vehicles (ROVs) or Autonomous Underwater
Vehicles (AUVs). In this context, the system could
provide real-time situational awareness to pilots by
highlighting points of interest and generating concise
mission summaries upon surfacing.
•Maritime Surveillance Buoy:The system can also
be deployed on a floating buoy, which guarantees a
continuouspowersourcetodelivermorefrequentand
higher-quality reports, process more queries, and uti-
lize hybrid and mission modes more often. Addition-
ally,theenergymanagementagentcanbeupgradedto
account for variable battery charging and discharging
cycles,allowingittooptimizetheactivityofboththe
sentinel nodes and the master node based on solar
patterns.
Futureworkwillfocusonrefiningthecurrentpipelineby
exploringefficientacousticembeddingtechniques—suchas
CLAPandSAMAudio—forfullymultimodaleventcorrela-
tion. Works on seismic detection models were already initi-
ated; however, developing reliable edge-deployable models
remains challenging due to training data scarcity and tight
microcontroller memory and compute constraints, Conse-
quently,thiscomponenthasbeenexcludedfromthecurrent
manuscriptpendingfurthervalidation.Additionally,weaim
tooptimizetheLLMinferencepipeline.Toeliminateimage
processingoverheadduringinference,wewillleveragepre-
storedCLIPembeddingsdirectlybyintroducingaprojector
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 23 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
that bridges the vision and Qwen embedding spaces, while
evaluating newly developed models under 4B parameters.
Finally, we plan to substitute general biological foundation
models with specialized marine domain architectures like
AquaticCLIP [2] to extract more granular aquatic environ-
mental features.
In conclusion, this work presents a viable and flexible
framework for intelligent underwater monitoring, specifi-
cally engineered to operate effectively under severe energy
and bandwidth restrictions. By leveraging local multimodal
RAG and adaptive identification, the system empowers re-
searchers to obtain actionable ecological insights from re-
mote deployments with minimal energy consumption and
communication overhead. Its modular design opens new
possibilities for diverse marine observation tasks, bridging
thegapbetweenrawdatacollectionandautomatedscientific
interpretation at the edge.
8. Acknowledgments
Thisworkwasconductedaspartofaninternshipfunded
by the French Research Institute for Exploitation of the Sea
(Ifremer).TheauthorsgratefullyacknowledgeIfremerforits
financial support and computational resources, specifically
the Datarmor HPC cluster that facilitated model training.
Additionally, we express our gratitude to the creators of the
WildFish dataset for granting access to their dataset, and to
Mathieu Léonardon for endorsing the preprint.
CRediT authorship contribution statement
Mohamed Amine Janati:Writing – review & editing,
Writing–originaldraft,Validation,Supervision,Conceptu-
alization,Methodology,Visualization,Software,Datacura-
tion,Resultanalysis.LaurentGautier:Writing–review&
editing,Supervision,Conceptualization,Projectadministra-
tion, Funding acquisition, Validation, Resources.Stéphane
Barbot:Writing – review & editing, Supervision, Con-
ceptualization, Project administration, Funding acquisition,
Validation, Resources.
References
[1] Abujabal, M., Saoud, L.S., Hussain, I., 2025. Fishdet-m: A unified
large-scalebenchmarkforrobustfishdetectionandclip-guidedmodel
selectionindiverseaquaticvisualdomains. doi:10.48550/arXiv.2507.
17859,arXiv:2507.17859.
[2] Alawode, B., Ganapathi, I.I., Javed, S., Werghi, N., Bennamoun, M.,
Mahmood, A., 2025. Aquaticclip: A vision-language foundation
modelforunderwatersceneanalysis. doi:10.48550/arXiv.2502.01785,
arXiv:2502.01785.
[3] Analog Devices, 2022a. MAX78002: Artificial Intelligence Micro-
controller with Low-Power Convolutional Neural Network Accel-
erator. Analog Devices. URL:https://www.analog.com/media/en/
technical-documentation/data-sheets/max78002.pdf. data sheet.
[4] Analog Devices, 2022b. MAX78002 Artificial Intelligence Micro-
controller with Low-Power Convolutional Neural Network Acceler-
ator Data Sheet. Technical Report. Analog Devices, Inc. URL:
https://www.analog.com/en/products/max78002.html.
[5] Australian Institute of Marine Science, 2020. OzFish Dataset -
Machine Learning Dataset for Baited Remote Underwater VideoStations. Available online:https://github.com/open-AIMS/ozfish.
doi:10.25845/5E28F062C5097. accessed: June 2026.
[6] Baumgartner,M.F.,Fratantoni,D.M.,Hurst,T.P.,Brown,M.W.,Cole,
T.V.N., Van Parijs, S.M., Johnson, M., 2013. Real-time reporting of
baleen whale passive acoustic detections from ocean gliders. The
JournaloftheAcousticalSocietyofAmerica134,1814–1823.doi:10.
1121/1.4816406.
[7] Cheng,T.,Song,L.,Ge,Y.,Liu,W.,Wang,X.,Shan,Y.,2024. Yolo-
world: Real-time open-vocabulary object detection. doi:10.48550/
arXiv.2401.17270,arXiv:2401.17270.
[8] Froese, R., Pauly, D., 2024. FishBase. World Wide Web electronic
publication. URL:https://www.fishbase.org. accessed: June 2026.
[9] Gong, Z., Wang, A.T., Huo, X., Haurum, J.B., Lowe, S.C., Taylor,
G.W., Chang, A.X., 2025. CLIBD: Bridging vision and genomics
forbiodiversitymonitoringatscale,in:ProceedingsoftheThirteenth
InternationalConferenceonLearningRepresentations. doi:10.48550/
arXiv.2405.17537.
[10] Gu, L., Stankovic, J.A., 2005. Radio-triggered wake-up for wireless
sensor networks. Real-Time Systems 29, 157–182. doi:10.1007/
s11241-005-6883-z.
[11] Irfan, M., Jiangbin, Z., Ali, S., Iqbal, M., Masood, Z., Hamid, U.,
2021. Deepship: An underwater acoustic benchmark dataset and a
separable convolution based autoencoder for classification. Expert
Systems with Applications 183, 115270. doi:10.1016/j.eswa.2021.
115270.
[12] Khan, F.F., Li, X., Temple, A.J., Elhoseiny, M., 2023. Fishnet:
A large-scale dataset and benchmark for fish recognition, detection,
and functional trait prediction, in: 2023 IEEE/CVF International
Conference on Computer Vision (ICCV), pp. 20439–20449. doi:10.
1109/ICCV51070.2023.01874.
[13] Laurent,G.,Barbot,S.,LeVourc’h,D.,2025. Syrene:Anunderwater
embeddedartificialintelligencecameraforinvasivefaunamonitoring.
Marine Technology Reporter Magazine 68, 40–42. URL:https:
//archimer.ifremer.fr/doc/00992/110327/.
[14] Lin,J.,Chen,W.M.,Lin,Y.,Cohn,J.,Gan,C.,Han,S.,2020.Mcunet:
Tiny deep learning on iot devices. doi:10.48550/arXiv.2007.10319,
arXiv:2007.10319.
[15] Liu,S.,Zeng,Z.,Ren,T.,Li,F.,Zhang,H.,Yang,J.,Jiang,Q.,Li,C.,
Yang,J.,Su,H.,Zhu,J.,Zhang,L.,2024. Groundingdino:Marrying
dino with grounded pre-training for open-set object detection, in:
Computer Vision – ECCV 2024: 18th European Conference, Mi-
lan, Italy, September 29–October 4, 2024, Proceedings, Part XLVII,
Springer-Verlag, Berlin, Heidelberg. p. 38–55. URL:https://doi.
org/10.1007/978-3-031-72970-6_3, doi:10.1007/978-3-031-72970-6_3.
[16] Mehrab, K.S., Maruf, M., Daw, A., Neog, A., Manogaran, H.B.,
Khurana, M., Feng, Z., Altintas, B., Bakis, Y., Campolongo, E.G.,
Thompson, M.J., Wang, X., Lapp, H., Berger-Wolf, T., Mabee, P.,
Bart, H., Chao, W.L., Dahdul, W.M., Karpatne, A., 2025. Fish-vista:
A multi-purpose dataset for understanding & identification of traits
from images, in: 2025 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pp. 24275–24285. doi:10.1109/
CVPR52734.2025.02261.
[17] Mekonnen, T., Harjula, E., Koskela, T., Ylianttila, M., 2017. sleepy-
cam: Power management mechanism for wireless video-surveillance
cameras,in:2017IEEEInternationalConferenceonCommunications
Workshops (ICC Workshops), pp. 91–96. doi:10.1109/ICCW.2017.
7962639.
[18] Millar, J., Huang, Y., Sethi, S., Haddadi, H., Madhavapeddy, A.,
2025. Benchmarking ultra-low-power𝜇npus. doi:10.48550/arXiv.
2503.22567,arXiv:2503.22567.
[19] Moosmann, J., Giordano, M., Vogt, C., Magno, M., 2023a. Tinyissi-
moYOLO: A quantized, low-memory footprint, tinyml object detec-
tion network for low power microcontrollers, in: 2023 IEEE 5th In-
ternationalConferenceonArtificialIntelligenceCircuitsandSystems
(AICAS). doi:10.1109/AICAS57966.2023.10168657.
[20] Moosmann, J., Giordano, M., Vogt, C., Magno, M., 2023b. Tinyis-
simoYOLO: A quantized, low-memory footprint, TinyML object
detection network for low power microcontrollers, in: 2023 IEEE
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 24 of 25

Energy Constrained Hierarchical Underwater Monitoring via Local Multi-Agent RAG
5th International Conference on Artificial Intelligence Circuits and
Systems (AICAS), IEEE. pp. 1–5. doi:10.1109/AICAS57966.2023.
10168657.
[21] Moosmann, J., Müller, H., Zimmerman, N., Rutishauser, G., Benini,
L., Magno, M., 2024. Flexible and fully quantized lightweight
TinyissimoYOLO for ultra-low-power edge systems. IEEE Access
12, 75093–75107. doi:10.1109/ACCESS.2024.3404878.
[22] Mots’oehli, M., Nikolaev, A., IGede, W.B., Lynham, J., Mous, P.J.,
Sadowski, P., 2024. Fishnet: Deep neural networks for low-cost fish
stock estimation, in: 2024 IEEE International Conference on Omni-
layer Intelligent Systems (COINS), IEEE. pp. 1–7. doi:10.1109/
coins61597.2024.10622134.
[23] Ortenzi, L., Aguzzi, J., Costa, C., Marini, S., D’Agostino, D., Thom-
sen, L., De Leo, F.C., Correa, P.V., Chatzievangelou, D., 2024.
Automated species classification and counting by deep-sea mobile
crawler platforms using YOLO. Ecological Informatics 82, 102788.
doi:10.1016/j.ecoinf.2024.102788.
[24] Palazzetti,L.,Giannetti,D.,Verolino,A.,Grasso,D.A.,Pinotti,C.M.,
Betti Sorbelli, F., 2025. AntPi: A raspberry pi based edge-cloud
system for real-time ant species detection using YOLO. Ecological
Informatics 91, 103383. doi:10.1016/j.ecoinf.2025.103383.
[25] Palomares, M.L.D., Pauly,D., 2024. SeaLifeBase. World Wide Web
electronicpublication. URL:https://www.sealifebase.org.accessed:
June 2026.
[26] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agar-
wal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G.,
Sutskever, I., 2021. Learning transferable visual models from nat-
ural language supervision, in: Proceedings of the 38th International
Conference on Machine Learning, PMLR. pp. 8748–8763.
[27] Ray, P.P., 2022. A review on tinyml: State-of-the-art and prospects.
Journal of King Saud University - Computer and Information Sci-
ences 34, 1595–1623. doi:10.1016/j.jksuci.2021.11.019.
[28] Song, Y., Lv, C., Zhu, K., 2026. FMRAG: Retrieval-augmented
multimodallargelanguagemodelsforfisheriesintelligence. Frontiers
in Marine Science 13, 1801835. doi:10.3389/fmars.2026.1801835.
[29] Stevens,S.,Wu,J.,Thompson,M.J.,Campolongo,E.G.,Song,C.H.,
Carlyn, D.E., Dong, L., Dahdul, W.M., Stewart, C., Berger-Wolf, T.,
Chao, W.L., Su, Y., 2024. BioCLIP: A vision foundation model
for the tree of life, in: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR). doi:10.48550/
arXiv.2311.18803.
[30] Velasco-Montero, D., Fernández-Berni, J., Carmona-Galán, R., San-
glas,A.,Palomares,F.,2024. ReliableandefficientintegrationofAI
into camera traps for smart wildlife monitoring based on continual
learning. Ecological Informatics 83, 102815. doi:10.1016/j.ecoinf.
2024.102815.
[31] Watkins, W.A., Fristrup, K., Daher, M.A., Howald, T.J., 1992.
SOUND Database of Marine Animal Vocalizations: Structure and
Operations. Technical Report. Woods Hole Oceanographic Institu-
tion. doi:10.1575/1912/854.
[32] Yang, C.H., Feuer, B., Jubery, Z., Deng, Z.K., Nakkab, A., Hasan,
M.Z.,Chiranjeevi,S.,Marshall,K.,Baishnab,N.,Singh,A.K.,Singh,
A., Sarkar, S., Merchant, N., Hegde, C., Ganapathysubramanian,
B., 2024. Biotrove: A large curated image dataset enabling ai for
biodiversity. arXiv preprint arXiv:2406.17720 doi:10.48550/arXiv.
2406.17720.
[33] Zhuang,P.,Wang,Y.,Qiao,Y.,2021. Wildfish++:Acomprehensive
fish benchmark for multimedia research. IEEE Transactions on
Multimedia 23, 3603–3617. doi:10.1109/TMM.2020.3028482.
Mohamed Amine JANATI et al.:Preprint submitted to ElsevierPage 25 of 25