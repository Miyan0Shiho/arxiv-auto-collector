# RS-RAG: Bridging Remote Sensing Imagery and Comprehensive Knowledge with a Multi-Modal Dataset and Retrieval-Augmented Generation Model

**Authors**: Congcong Wen, Yiting Lin, Xiaokang Qu, Nan Li, Yong Liao, Hui Lin, Xiang Li

**Published**: 2025-04-07 12:13:43

**PDF URL**: [http://arxiv.org/pdf/2504.04988v1](http://arxiv.org/pdf/2504.04988v1)

## Abstract
Recent progress in VLMs has demonstrated impressive capabilities across a
variety of tasks in the natural image domain. Motivated by these advancements,
the remote sensing community has begun to adopt VLMs for remote sensing
vision-language tasks, including scene understanding, image captioning, and
visual question answering. However, existing remote sensing VLMs typically rely
on closed-set scene understanding and focus on generic scene descriptions, yet
lack the ability to incorporate external knowledge. This limitation hinders
their capacity for semantic reasoning over complex or context-dependent queries
that involve domain-specific or world knowledge. To address these challenges,
we first introduced a multimodal Remote Sensing World Knowledge (RSWK) dataset,
which comprises high-resolution satellite imagery and detailed textual
descriptions for 14,141 well-known landmarks from 175 countries, integrating
both remote sensing domain knowledge and broader world knowledge. Building upon
this dataset, we proposed a novel Remote Sensing Retrieval-Augmented Generation
(RS-RAG) framework, which consists of two key components. The Multi-Modal
Knowledge Vector Database Construction module encodes remote sensing imagery
and associated textual knowledge into a unified vector space. The Knowledge
Retrieval and Response Generation module retrieves and re-ranks relevant
knowledge based on image and/or text queries, and incorporates the retrieved
content into a knowledge-augmented prompt to guide the VLM in producing
contextually grounded responses. We validated the effectiveness of our approach
on three representative vision-language tasks, including image captioning,
image classification, and visual question answering, where RS-RAG significantly
outperformed state-of-the-art baselines.

## Full Text


<!-- PDF content starts -->

1
RS-RAG: Bridging Remote Sensing Imagery and
Comprehensive Knowledge with a Multi-Modal
Dataset and Retrieval-Augmented Generation Model
Congcong Wen Member, IEEE , Yiting Lin, Xiaokang Qu, Nan Li, Yong Liao, Hui Lin, and Xiang Li
Abstract —Recent progress in Vision-Language Models (VLMs)
has demonstrated impressive capabilities across a variety of
tasks in the natural image domain. Motivated by these ad-
vancements, the remote sensing community has begun to adopt
VLMs for remote sensing vision-language tasks, including scene
understanding, image captioning, and visual question answering.
However, existing remote sensing VLMs typically rely on closed-
set scene understanding and focus on generic scene descriptions,
yet lack the ability to incorporate external knowledge. This
limitation hinders their capacity for semantic reasoning over
complex or context-dependent queries that involve domain-
specific or world knowledge. To address these challenges, we
first introduced a multimodal Remote Sensing World Knowledge
(RSWK) dataset, which comprises high-resolution satellite im-
agery and detailed textual descriptions for 14,141 well-known
landmarks from 175 countries, integrating both remote sensing
domain knowledge and broader world knowledge. Building upon
this dataset, we proposed a novel Remote Sensing Retrieval-
Augmented Generation (RS-RAG) framework, which consists
of two key components. The Multi-Modal Knowledge Vector
Database Construction module encodes remote sensing imagery
and associated textual knowledge into a unified vector space.
The Knowledge Retrieval and Response Generation module
retrieves and re-ranks relevant knowledge based on image and/or
text queries, and incorporates the retrieved content into a
knowledge-augmented prompt to guide the VLM in producing
contextually grounded responses. We validated the effectiveness
of our approach on three representative vision-language tasks,
including image captioning, image classification, and visual ques-
tion answering, where RS-RAG significantly outperformed state-
of-the-art baselines. By bridging remote sensing imagery and
comprehensive knowledge, RS-RAG empowers remote sensing
VLMs with enhanced contextual reasoning, enabling them to
generate more accurate, informative, and semantically grounded
outputs across a wide range of tasks.
Index Terms —Remote Sensing, Vision Language Model,
Retrieval-Augmented Generation, World Knowledge
This work was partly supported by the Beijing Nova Program under Grant
2024124 and the National Natural Science Foundation of China under Grant
U24B20177. (Congcong Wen and Yiting Lin contributed equally to this work).
(Corresponding author: Hui Lin) .
Congcong Wen, Yiting Lin, Xiaokang Qu and Yong Liao are with School
of Cyber Science and Technology, University of Science and Technology of
China, Anhui, 230026, China. Congcong Wen is also with the Department of
Electrical and Computer Engineering, New York University Abu Dhabi, Abu
Dhabi, UAE. (e-mail: wencc1208@gmail.com, linyiting@mail.ustc.edu.cn,
xkqu@mail.ustc.edu.cn and yliao@ustc.edu.cn)
Nan Li and Hui Lin are with China Academy of Electronics and Infor-
mation Technology, Beijing 100846, China. (e-mail: linhui@whu.edu.cn and
nli2014@lzu.edu.cn)
Xiang Li is with the Department of Computer Science at the University of
Reading, Reading RG6 6AH, UK. (e-mail: xiangli92@ieee.org)I. I NTRODUCTION
Remote sensing imagery, as a critical source of informa-
tion for Earth observation and monitoring, plays an essential
role in urban planning [1], agricultural assessment [2], and
environmental protection [3]. However, as remote sensing
technology advances, the scale and complexity of imagery
data have rapidly increased, making it increasingly difficult
for traditional manual analysis or image processing methods to
meet practical demands. Deep learning methods [4], [5] have
significantly improved the accuracy and efficiency of tasks like
classification, segmentation and object detection by automati-
cally extracting features from vast amounts of remote sensing
data. While these methods have made notable progress, most
deep learning models rely predominantly on single-modal
visual information, lacking deep semantic understanding of
image content. This limitation results in reduced generalization
and adaptability, particularly for tasks that require in-depth
semantic analysis and comprehensive scene understanding.
The emergence of Vision-Language Models (VLMs) [6],
[7], [8], [9], [10], [11] offers a novel solution for the semantic
analysis of remote sensing data. By leveraging multimodal fu-
sion techniques, VLMs combine visual features with language
information to automatically generate descriptive insights for
remote sensing imagery. This semantic enhancement improves
image classification and object detection performance while
enabling the transformation of recognition results into nat-
ural language descriptions, making them more interpretable
and accessible for various applications. Additionally, VLMs
perform well even in weakly supervised or zero-shot learning
scenarios, providing reliable analysis with limited labeled data
and thus reducing dependency on extensive data annotations.
This cross-modal integration not only enhances the model’s
cognitive ability to interpret remote sensing imagery but also
facilitates detailed semantic descriptions of complex scenes,
paving the way for broader intelligent applications in remote
sensing. However, existing remote sensing VLMs primarily
focus on identifying image features and providing basic scene
descriptions, lacking deeper background understanding of the
objects within the images. Particularly when it comes to
rich semantic information requiring remote sensing domain
expertise or other general world knowledge, such as historical,
cultural, and social contexts, these VLM models often struggle
to provide comprehensive contextual support.
To address this issue, we first introduce the Remote Sensing
World Knowledge (RSWK) dataset, a multimodal remote sens-arXiv:2504.04988v1  [cs.CV]  7 Apr 2025

2
ing dataset that contains high-resolution imagery and natural
language descriptions for approximately 14,141 well-known
locations worldwide from 175 conuties. Unlike most existing
remote sensing vision-language datasets that only provide
basic descriptions of current scenes, our RSWK dataset will
include richer remote sensing domain expertise and world
knowledge about the objects within these scenes. For instance,
from a remote sensing perspective, it will provide information
on surface reflectance, spectral indices, and atmospheric. From
a world knowledge perspective, it will include historical back-
ground, cultural significance, construction period, and major
events. This combination of remote sensing expertise and
world knowledge will not only enhance the RSWK dataset’s
utility for visual analysis of remote sensing images, but also
provide the model with deeper semantic context, overcoming
the limitations of traditional datasets and enabling remote
sensing VLMs to perform more complex cognitive tasks.
Furthermore, our dataset incorporates historical, cultural, and
social backgrounds from various countries and regions, allow-
ing VLM models to be trained across diverse geographical
and cultural contexts, thereby improving their generalization
ability and understanding of different cultural settings.
Furthermore, to effectively leverage the comprehensive
information provided by the RSWK dataset, we propose
the Remote Sensing Retrieval-Augmented Generation (RS-
RAG) model. RS-RAG is designed to enhance the capacity
of vision-language models to generate contextually enriched
and knowledge-grounded responses for remote sensing im-
agery. It operates by integrating external knowledge, both
domain-specific and general world knowledge, retrieved from
a multimodal knowledge base constructed using the RSWK
dataset. The model consists of two main components: (1)
the Multi-Modal Knowledge Vector Database Construction
module, which encodes satellite imagery and textual descrip-
tions into a shared embedding space using unified image
and text encoders to enable efficient cross-modal retrieval;
and (2) the Knowledge Retrieval and Response Generation
module, which retrieves top-ranked knowledge entries based
on image and text queries, re-ranks them via a fused sim-
ilarity score, and incorporates the selected content into a
knowledge-augmented prompt. This prompt is then passed to
the vision-language model, enabling it to generate responses
that go beyond semantic understanding and reflect deeper
background knowledge. By coupling visual input with relevant
contextual information, RS-RAG significantly improves the
interpretability and accuracy of vision-language outputs, par-
ticularly for complex queries involving geospatial, historical,
or environmental reasoning. To assess the effectiveness of our
proposed model, we construct a lightweight benchmark and
conduct comprehensive experiments on three representative
tasks: image captioning, image classification, and visual ques-
tion answering. Results on these tasks demonstrate the model’s
ability to produce accurate, context-rich descriptions, deliver
semantically informed scene classifications, and provide pre-
cise answers to knowledge-intensive queries by leveraging
both visual and textual modalities. Through these findings, RS-
RAG demonstrates its potential to substantially advance the
capabilities of vision-language models in remote sensing, ef-fectively bridging the gap between imagery and comprehensive
contextual knowledge. Our main contributions are summarized
as follows:
•We construct the Remote Sensing World Knowledge
(RSWK) dataset, a large-scale multimodal benchmark
containing 14,141 high-resolution remote sensing images
and rich textual descriptions of globally recognized land-
marks from 175 countries. The descriptions incorporate
both domain-specific knowledge (e.g., land use, tempera-
ture, wind direction) and general world knowledge (e.g.,
historical, cultural, and societal context).
•We propose RS-RAG, a novel Retrieval-Augmented Gen-
eration framework tailored for remote sensing vision-
language tasks. RS-RAG retrieves semantically relevant
knowledge from a multimodal vector database and in-
tegrates it with the input via knowledge-conditioned
prompt construction, significantly enhancing contextual
reasoning capabilities.
•We design a lightweight benchmark for evaluating re-
mote sensing vision-language models across three core
tasks: image captioning, image classification, and visual
question answering. This benchmark enables systematic
assessment of both semantic understanding and deeper
knowledge-grounded reasoning.
•Extensive experiments demonstrate that RS-RAG consis-
tently outperforms state-of-the-art vision-language mod-
els across all tasks, particularly on queries requiring
external world knowledge. These results highlight the
effectiveness of RS-RAG in bridging remote sensing im-
agery with structured knowledge, and point to promising
future directions for research on remote sensing vision-
language models.
II. R ELATED WORK
A. Remote Sensing Multimodal Datasets
Several multimodal datasets [12], [13], [14], [15], [16] have
been developed to bridge the gap between vision and language
in the remote sensing domain. UCM Captions [17], Sydney
Captions [17], and RSICD [18] are among the earliest datasets
that provide textual descriptions for remote sensing images.
Each image in these datasets is paired with five relatively
simple human-written sentences, offering only a basic level of
semantic information. To enhance the quality and richness of
textual annotations, RSGPT [6] recently introduced RSICap,
a high-quality image captioning dataset with detailed human-
annotated descriptions of aerial scenes. RSICap serves as
a valuable resource for fine-tuning and developing domain-
specific vision-language models in remote sensing. Beyond
manual annotation, researchers have explored the construction
of large-scale datasets using automatic or hybrid approaches
applied to existing data sources. For instance, RS5M [19]
was created by aggregating 11 publicly available image–text
paired datasets along with three large-scale class-level labeled
datasets. Captions were generated using BLIP-2, resulting
in a diverse and large-scale dataset suitable for training
foundational multimodal models. Similarly, RemoteCLIP [20]

3
Fig. 1. The construction process of the Remote Sensing World Knowledge (RSWK) dataset begins with collecting landmark data from around the world,
followed by extracting geographic information to pinpoint precise location coordinates. Using these coordinates, remote sensing images are acquired, which are
further standardized through image processing techniques. Corresponding remote sensing expert knowledge, such as surface temperature, climate, atmospheric
conditions, and spectral coefficients, is also included. Additionally, world knowledge is retrieved from online resources, providing detailed background
information about the landmarks, including historical context, cultural significance, and major events. This combined information is structured into organized
attributes to align image and text data, forming a final multimodal dataset. The resulting RSWK dataset integrates high-resolution images with extensive
remote sensing and world knowledge, enabling advanced semantic understanding in remote sensing applications.
compiled a large-scale dataset by integrating 10 object detec-
tion datasets, 4 semantic segmentation datasets, and 3 remote
sensing image–text datasets, enabling contrastive pretraining
for cross-modal alignment. In addition, GeoChat introduced
the RS Multimodal Instruction Dataset, which incorporates
heterogeneous data sources, including three object detection
datasets, one scene classification dataset, and a visual question
answering dataset focused on flood detection. More recently,
FedRSCLIP [21] proposed a new multimodal remote sensing
dataset specifically designed for federated learning scenarios,
further expanding the applicability of vision-language research
in distributed settings. These efforts collectively advance the
development of remote sensing VLMs by providing diverse,
rich, and large-scale multimodal data resources tailored to
various downstream tasks.
B. Remote Sensing Multimodal Model
With the growing availability of remote sensing multi-
modal datasets, a number of vision-language models [6],
[7], [8], [9], [10], [11] have been proposed to enhance the
understanding and interpretation of aerial imagery through
natural language. [22] provides the first comprehensive review
of vision-language models in remote sensing, systematically
summarizing tasks, datasets, and methods, and identifying key
challenges and future directions. RSCLIP [23] is a pioneering
vision-language model designed for remote sensing scene clas-
sification, which leverages contrastive vision-language super-
vision and incorporates a pseudo-labeling technique along with
a curriculum learning strategy to improve zero-shot classifica-
tion performance through multi-stage fine-tuning. RSGPT [6]represents one of the earliest attempts to build a generative
pretrained model for remote sensing. It fine-tunes only the
Q-Former and a linear projection layer on the proposed RSI-
Cap dataset, achieving notable improvements in both image
captioning and visual question answering tasks. Similarly,
GeoChat [24] adapts the LLaV A-1.5 architecture and fine-
tunes it on its proposed remote sensing multimodal instruction-
following dataset, offering multi-task conversational capabili-
ties grounded in high-resolution satellite imagery. In addition,
to address the common issue of hallucination in remote sensing
vision-language models, a Helpful and Honest Remote Sensing
Vision-Language Model [8], named H2RSVLM, is proposed
and fine-tuned on the RSSA dataset, the first dataset specif-
ically designed to enhance self-awareness in remote sensing
VLMs. More recently, RSMoE [25] was proposed as the first
Mixture-of-Experts-based VLM tailored for remote sensing. It
features a novel instruction router that dynamically dispatches
tasks to multiple lightweight expert LLMs, allowing each
expert to specialize in a specific subset of tasks.
III. D ATASET CONSTRUCTION
Most existing remote sensing VLM datasets focus primarily
on basic descriptions of objects in a scene, typically provid-
ing only information about the objects present. While these
datasets serve a purpose in supporting fundamental tasks such
as scene classification and image captioning, they fall short in
applications requiring complex semantic understanding or con-
textual awareness. To address this limitation, in our full paper,
we would like to propose the Remote Sensing World Knowl-
edge (RSWK) dataset, which encompasses high-resolution

4
Name Sydney Opera House
Category Theater
Area Bennelong Point, Sydney
Location 33°51′25″S 151 °12′55″E
AddressBennelong Point, Sydney, New South Wales, 
Australia
Physical Area 1.8 hectares (4.4 acres)
Construction 
Period1959 –1973
Historical 
BackgroundThe Sydney Opera House is a multi -venue 
performing arts centre in Sydney, New South 
Wales, Australia. Designed by Danish architect Jørn
Utzon and completed by an Australian architectural 
team headed by Peter Hall, the building was 
formally opened by Queen Elizabeth II on 20 
October 1973. It is widely regarded as one of the 
world's most famous and distinctive buildings and 
a masterpiece of 20th -century architecture.
Major Events• Formal Opening (1973): The Sydney Opera House 
was formally opened by Queen Elizabeth II on 20 
October 1973, featuring a performance of 
Beethoven's Symphony No. 9 and fireworks.
• UNESCO Designation (2007): The Sydney Opera 
House was designated a UNESCO World Heritage 
Site in 2007, recognizing its cultural and 
architectural significance.
Architectural 
Characteristi
csThe Sydney Opera House features a modern 
expressionist design with large precast concrete 
"shells" forming the roofs. It rests on 588 concrete 
piers and uses over 2,000 pre -cast sections. 
Exterior cladding includes pink granite; interiors 
use white birch plywood and glulam.
Cultural 
SignificanceThe Sydney Opera House is a UNESCO World 
Heritage Site and national icon of Australia. It 
appears on multiple heritage registers and was a 
finalist in the New 7 Wonders campaign.
Primary 
FunctionPerforming arts centre
Notable 
VisitorsQueen Elizabeth II
DetailsThe Opera House hosts over 1,800 performances 
yearly with 1.4 million+ attendees. It is managed by 
the Sydney Opera House Trust and offers guided 
and backstage tours. Facilities include multiple 
venues, studios, shops, restaurants, and bars.Albedo 0.08470577
Albnirdf 0.154914677
Albnirdr 0.125447109
Albvisdf 0.046561256
Albvisdr 0.032145903
Emis 0.993293047
Evland 6.77e -05
Gwetprof 0.47887823
Lc_type1 17
Lc_type2 0
Lc_type3 0
Lc_type4 0
Lc_type5 0
LST_Day_1km Nan
LST_Night_1km Nan
Prectotland 7.51e -08
Saa Nan
Slp 100563.0
Smland 0.0
Speed 3.065944433
Sr_b1 16116.0
Sr_b2 16175.0
Sr_b3 16490.0
Sr_b4 16883.0
Sr_b5 19245.0
Sr_b6 17615.0
Sr_b7 16327.0
Ts 298.9092712
Ulml 2.087645292
Vaa Nan
Vlml -0.902604997
Vza Nan
Mean_2m_air_temperature 298.4058533
Mean_sea_level_pressure 100645.2734
Surface_pressure 99936.96875
Total_precipitation 0.014615823a)Global distribution of landmarks used in the RSWK dataset .
b)Statistics of landmark distribution across countries and categories in the RSWK dataset. c)An example from the RSWK dataset showing the Sydney Opera House .Domain KnowledgeWorld Knowledge Imagery
Fig. 2. Overview of the RSWK dataset. (a) Global distribution of landmarks used in the dataset, with color indicating the number of landmarks per country.
(b) Statistical summaries of landmark counts across the top 100 countries (left) and the top 15 most frequent landmark categories (right). (c) A specific
example from the RSWK dataset, showcasing the Sydney Opera House, including its satellite imagery, remote sensing domain knowledge, and structured
world knowledge.
remote sensing imagery of well-known locations worldwide,
along with domain knowledge and world knowledge described
in natural language. This dataset not only fills the gap in
knowledge depth and breadth found in current remote sensing
datasets but also provides a new foundation for advancing
remote sensing technology toward intelligent applications.
The core value of remote sensing data lies in its ability to
deliver rich geographic and spatial distribution information.
By integrating high-resolution imagery from across the globe
with detailed domain knowledge and world knowledge, the
RSWK dataset extends this value, expanding the application
of remote sensing data from traditional foundational tasks to
complex scenarios that require deeper semantic understanding.
A. Data Processing
Fig. 1 illustrates the end-to-end pipeline for constructing the
Remote Sensing World Knowledge (RSWK) dataset, which
aims to bridge the gap between remote sensing domain
knowledge and encyclopedic world knowledge of globally dis-
tributed landmarks. This pipeline is designed to automatically
curate and align multimodal information from global sources,
integrating high-resolution remote sensing imagery, remote
sensing expert knowledge, and contextual world knowledge.
The process begins with GPT-4o [26], a state-of-the-art large
language model, which is employed to generate a compre-
hensive list of globally recognized landmarks across diverse
countries and regions, ensuring broad cultural and geographic
coverage that includes both natural and man-made sites ofhistorical and societal relevance. To acquire world knowledge,
we utilize the Wikipedia API to extract descriptive information
for each landmark, such as historical background, cultural
relevance, architectural details, and notable events, with the
complete set of fields shown in Table I. Post-processing is
performed using DeepSeek [27]to remove low-quality or irrel-
evant entries. In parallel, the Google Geocoding API is used
to obtain precise geographic coordinates for each location; to
resolve potential ambiguities in place names, we implement
a validation mechanism that cross-references multiple sources
and applies regional constraints. High-resolution satellite im-
agery is retrieved from the ArcGIS Tile Map Service, with
spatial resolutions ranging from 0.6m to 0.15m, followed by
normalization, cropping, and resizing to ensure consistency
in image quality and dimensions. Concurrently, remote sens-
ing expert knowledge is derived using Google Earth Engine
(GEE) [28], as detailed in Table II. This includes a wide range
of satellite-derived geophysical and meteorological variables
such as land surface temperature, surface albedo, land cover
classification, vegetation indices, and precipitation, which are
sourced from authoritative datasets like MODIS, Landsat, and
ERA5. By integrating these three complementary modalities,
remote sensing imagery, remote sensing expert knowledge, and
contextual world knowledge, the RSWK dataset provides a
rich foundation for multimodal learning in remote sensing,
enabling downstream tasks such as image captioning, image
classification, and visual question answering.

5
TABLE I
WORLD KNOWLEDGE FIELDS AND DESCRIPTIONS FOR LANDMARK IN
THE RSWK DATASET .
Field Name Description
Name The official or commonly known name of the landmark.
Category The type of the landmark.
Area The general geographical region or country where the land-
mark is located.
Location Latitude and longitude coordinates of the landmark.
Address Full postal or descriptive address of the landmark.
Physical_Area The spatial footprint or size of the landmark.
Construction_Period The years or range of years during which the landmark was
constructed.
Historical_Background Historical context, including origin, founding date, and devel-
opment over time.
Major_Events Key historical or cultural events that occurred at the landmark.
Architectural_Characteristics Notable features such as design style, construction materials,
and structure.
Cultural_Significance The cultural, symbolic, or religious importance of the land-
mark.
Primary_Function The main use or role of the landmark (e.g., tourist attraction,
government building).
Notable_Visitors Famous individuals or groups who have visited the site.
Details Additional facts or interesting information not captured by
other fields.
TABLE II
REMOTE SENSING DOMAIN KNOWLEDGE FIELDS , DESCRIPTIONS ,AND
SOURCE FOR LANDMARK IN THE RSWK DATASET .
Field Name Description Source
Albedo Surface albedo, representing the reflectivity
of the Earth’s surfaceMERRA-2 (M2T1NXRAD)
Albnirdf Near-infrared diffuse surface albedo MERRA-2 (M2T1NXRAD)
Albnirdr Near-infrared direct surface albedo MERRA-2 (M2T1NXRAD)
Albvisdf Visible light diffuse albedo MERRA-2 (M2T1NXRAD)
Albvisdr Visible light direct albedo MERRA-2 (M2T1NXRAD)
Emis Surface emissivity, related to thermal radia-
tionMERRA-2 (M2T1NXRAD)
Evland Evaporation land ( kg/m2/s) MERRA-2 (M2T1NXLND)
Gwetprof Profile soil moisture averaged over depth MERRA-2 (M2T1NXLND)
LC_type1 Land cover type (IGBP classification) MODIS (MCD12Q1.061)
LC_type2 Land cover type (UMD classification) MODIS (MCD12Q1.061)
LC_type3 Land cover type (LAI classification) MODIS (MCD12Q1.061)
LC_type4 Land cover type (BGC classification) MODIS (MCD12Q1.061)
LC_type5 Land cover type (Plant Functional Types
classification)MODIS (MCD12Q1.061)
LST_Day_1km Daytime land surface temperature (°C) MODIS (MOD11A1.061)
LST_Night_1km Nighttime land surface temperature (°C) MODIS (MOD11A1.061)
Prectotland Total land precipitation ( kg/m2/s) MERRA-2 (M2T1NXLND)
Saa Sun Azimuth Angle (°) HLSL30 (HLS-2)
Slp Sea level pressure (Pa) MERRA-2 (M2T1NXSLV)
Smland Snowmelt flux land ( kg/m2/s) MERRA-2 (M2T1NXLND)
Speed Surface wind speed ( m/s) MERRA-2 (M2T1NXFLX)
Sr_b1 Band 1 (ultra blue, coastal aerosol) surface
reflectanceLandsat 9
Sr_b2 Band 2 (blue) surface reflectance Landsat 9
Sr_b3 Band 3 (green) surface reflectance Landsat 9
Sr_b4 Band 4 (red) surface reflectance Landsat 9
Sr_b5 Band 5 (near infrared) surface reflectance Landsat 9
Sr_b6 Band 6 (shortwave infrared 1) surface re-
flectanceLandsat 9
Sr_b7 Band 7 (shortwave infrared 2) surface re-
flectanceLandsat 9
Ts Surface skin temperature (°C) MERRA-2 (M2T1NXSLV)
Ulml Surface eastward wind ( m/s) MERRA-2 (M2T1NXFLX)
Vaa View Azimuth Angle (°) HLSL30 (HLS-2)
Vlml Surface northward wind speed ( m/s) MERRA-2 (M2T1NXFLX)
Vza View Zenith Angle (°) HLSL30 (HLS-2)
Mean_2m_air_temperature Average air temperature at 2m height (°C) ERA5 (Daily Aggregates)
Mean_sea_level_pressure Mean sea level pressure (Pa) ERA5 Daily Aggregates
Surface_pressure Surface pressure (Pa) ERA5 (Daily Aggregates)
Total_precipitation Total precipitation (m) ERA5 (Daily Aggregates)
B. Data Statistics
Through the above data construction pipeline, the RSWK
dataset successfully collected a total of 14,141 landmark
instances from 175 countries, each accompanied by high-
resolution satellite imagery, domain-specific remote sensing
attributes, and structured world knowledge descriptions. The
global spatial distribution of the landmarks is visualized in
Fig. 2(a), demonstrating wide coverage across all major conti-
nents. Notably, the dataset includes landmarks from a diverse
range of regions, effectively covering most major countries
worldwide. To provide a more detailed view of the dataset
composition, we present the distribution of landmarks across
the top 100 countries sorted by landmark count in Fig. 2(b,
left). It can be observed that countries such as the UnitedStates, the United Kingdom, China, and Japan contribute the
largest number of landmarks. This reflects both the cultural
richness and the degree of documentation available for land-
marks in these regions. In addition, Fig. 2(b, right) shows
the frequency distribution of the top 15 landmark categories,
including parks, museums, natural scenery, universities, and
historic sites. The category distribution is relatively balanced,
highlighting the diverse types of landmarks captured in the
dataset and ensuring a rich set of semantic concepts for
downstream tasks. To provide an intuitive example of how
the data is structured, we present the Sydney Opera House in
Fig. 2(c) as a representative landmark. The figure illustrates
the three core components of each RSWK entry: the satellite
imagery, the domain knowledge (e.g., albedo, emissivity, land
cover type, and meteorological variables), and the structured
world knowledge (e.g., historical background, architectural
characteristics, and cultural significance). This tri-modal rep-
resentation demonstrates the depth and richness of the dataset,
supporting a wide range of geospatial understanding and
vision-language reasoning tasks.
IV. M ETHODOLOGY
To bridge the semantic gap between remote sensing imagery
and comprehensive external knowledge, we propose RS-RAG,
a Retrieval-Augmented Generation framework designed to in-
tegrate both domain-specific and world knowledge into vision-
language reasoning. As illustrated in Fig. 3, RS-RAG consists
of two main components: the Multi-Modal Knowledge Vector
Database Construction module, which encodes remote sensing
imagery and textual knowledge into a unified embedding
space; and the Knowledge Retrieval and Response Gener-
ation module, which retrieves and fuses the most relevant
knowledge to support downstream tasks. By conditioning the
vision-language model on retrieved context, RS-RAG enables
knowledge-grounded understanding for diverse applications
such as image captioning, scene classification, and visual
question answering.
A. Problem Formulation
VLMs are designed to generate natural language outputs
conditioned on multimodal inputs, typically a visual obser-
vation and a textual prompt. Let qIdenote the input image
(e.g., a remote sensing image), and qTthe associated textual
prompt (e.g., a question or instruction). A conventional VLM
seeks to generate a natural language response ˆyby modeling
the conditional probability distribution over output space and
selecting the most probable response. Formally, this can be
expressed as:
ˆy= arg max
yP(y|qT, qI;θVLM), (1)
where θVLM represents the parameters of the VLM, ˆycan
represent an image caption, a classification label, or an answer
to a visual question. This closed-form generation framework
assumes that all necessary information for reasoning is either
visually grounded in the input image qIor implicitly encoded
within the model parameters θVLM. While this assumption
often holds for natural image datasets, it becomes problematic

6
Text
EncoderText Embedding
Chunking
Domain andWorld KnowledgeEncoding
Image
EncoderImage
EmbeddingRemote Sensing Image
Encoding
Image Captioning Image Classification Visual Question AnsweringKnowledge -Conditioned 
Context FusionKnowledge -Augmented Prompt 
ConstructionVision -Language 
ModelKnowledge -Grounded Response 
Generation a)Multi -Modal Knowledge V ector  Database Construction b)Knowledge Retrieval and Response Generation
Vector Database
Response:Top-τImages
Retrieval Re-rankingTop-K TextsText
Encoder
Which  famous  landmark  
or city is shown  in this 
satellite  image?
Image
EncoderImage
Embedding
Text Embedding
The Sydney  Opera  House is a 
multi -venue performing  arts 
centre inSydney,  New South  
Wales,  Australia . Located  on 
the foreshore  of Sydney  
Harbour , it is widely  regarded  
as one of the world's  most 
famous  and distinctive  
buildings,  and a masterpiece  of 
20th-century  architecture .
The satellite image shows the Sydney 
Opera House, a world -famous landmark 
located in Sydney, Australia.
Query:
Top-τTexts
User:
What is the primary land cover type in this remote sensing image
based onspectral coefficients?
RS-RAG :
Based  on the analysis  of NDVI,  NDWI,  and NDBI,  we can 
identify  multiple  land cover  types  in this area. NDWI  reveals  
the presence  of water  bodies,  particularly  in the region  
corresponding  to harbour . NDBI  highlights  built-up areas,  
capturing  the bridge  structure  and surrounding  urban  
infrastructure . Meanwhile,  NDVI  indicates  patches  of 
vegetation,  primarily  in green  spaces  near the base of the 
bridge  and surrounding  parks . 
User:
What is the name and location of the building in this image?
RS-RAG :
The building  in this remote  sensing  image  is the Sydney  
Harbour  Bridge,  located  in Sydney,  Australia .
User:
What is the category of this remote sensing image? Answer the 
question using a single word or phrase. 
RS-RAG :
Stadium .
User:
What is the category of this remote sensing image? Answer the 
question using a single word or phrase. 
RS-RAG :
Airport .
xxxxxUser:
Please provide detailed descriptions for this remote sensing 
image.
RS-RAG :
This is a remote  sensing  image  of the Eiffel  Tower  and its 
surrounding  area,  located  in the 7th arrondissement  in western  
Paris,  at the northwest  end of the Champ  de Mars,  with an 
elevation  of 33 meters . The image  displays  a region  
characterized  by urban  land cover,  with nearby  grasslands,  
roads,  and water . The Seine  River  is visible  in the upper  left 
corner  of the image . Standing  at 324 meters,  the Eiffel  Tower  
is one of the tallest  structures  in Paris . Built  between  1887  and 
1889  for the 1889  Exposition  Universelle  (World’s  Fair)  held 
to celebrate  the 100th anniversary  of the French  Revolution,  
the tower  was designed  by engineer  Gustave  Eiffel . Its unique  
iron lattice  structure  has since  become  an iconic  symbol  of 
France . The surrounding  architecture  reflects  Paris‘s  historical  
and cultural  heritage,  blending  classical  and modern  structures . 
The open  spaces  around  the Eiffel  Tower,  including  the 
Champ  de Mars,  serve  as popular  spots  for tourists,  offering  
scenic  green  areas  amidst  the dense  cityscape .
Fig. 3. Overview of the proposed Remote Sensing Retrieval-Augmented Generation (RS-RAG) model. It consists of two main processes: (a) The Multi-Modal
Knowledge Vector Database Construction module encodes remote sensing imagery and domain/world knowledge into a unified vector space via image and text
encoders, enabling efficient cross-modal retrieval. (b) The Knowledge Retrieval and Response Generation module retrieves top-k relevant knowledge based
on image and/or textual queries, and re-ranks the results for better relevance. Retrieved knowledge is fused into the prompt through Knowledge-Conditioned
Context Fusion, guiding the Vision-Language Model (VLM) to generate Knowledge-Grounded Responses. The RS-RAG model supports multiple downstream
tasks such as Image Captioning, Image Classification, and Visual Question Answering, as demonstrated in the bottom section.
in the domain of remote sensing. Remote sensing images
typically capture large-scale scenes, such as entire cities or
extensive geographic regions, where accurate interpretation of-
ten relies on understanding cultural, historical, or geographical
significance. Such knowledge is rarely discernible from visual
features and is not explicitly modeled in conventional VLMs.
To address this limitation, we extend the standard VLM
formulation by incorporating external, query-relevant knowl-
edge. Specifically, we adopt a Retrieval-Augmented Genera-
tion (RAG) framework, wherein a set of top- krelevant knowl-
edge snippets Ris retrieved from an external corpus based on
the multimodal similarity between the input image–text pair
(qI, qT)and the knowledge index. The retrieved context Ris
then integrated with the original inputs to guide the response
generation process. Formally, the objective becomes:
ˆy= arg max
yP(y|qT, qI,R;θVLM), (2)
where the additional context Rallows the model to produce
more informative and contextually grounded outputs. This
open-book generation paradigm is particularly well-suited
for remote sensing applications, where high-level semanticunderstanding often depends on both domain-specific knowl-
edge (e.g., land use categories) and broader world knowledge
(e.g., cultural or geopolitical significance)—information that
is typically absent from raw pixel data alone.
B. Multi-Modal Knowledge Vector Database Construction
To enable retrieval-augmented generation, we construct a
Multi-Modal Knowledge Vector Database (MKVD) by encod-
ing the RSWK dataset, which contains high-resolution remote
sensing images Iiand their paired textual descriptions Ti.
These data are transformed into dense embeddings using CLIP
and stored in a shared semantic space to support efficient and
flexible cross-modal retrieval. We adopt CLIP as a unified
encoder consisting of a image encoder fI(·)and a text encoder
fT(·), each mapping input into a shared embedding space Rd.
Each image Iiis encoded into a visual embedding:
vi=fI(Ii)∈Rd. (3)
In parallel, the corresponding textual document Ti
is segmented into misemantically coherent chunks

7
{Ti,1, Ti,2, . . . , T i,m i}, and each chunk is encoded using
the text encoder:
ti,j=fT(Ti,j)∈Rd. (4)
The resulting image embeddings {vi}and text embeddings
{ti,j}are indexed in Qdrant, a high-performance vector
database optimized for approximate nearest neighbor (ANN)
search. To organize the data, embeddings are stored in two
separate collections: Dimage for image embeddings and Dtextfor
text embeddings. Each image-text pair is linked via a unique
identifier IDi. Specifically, each vi∈ D image is associated with
a set{ti,j}mi
j=1⊂ D text, which encodes domain-specific and
general world knowledge about the same location or object.
Metadata such as raw textual descriptions, image paths, and
geospatial attributes are stored alongside each entry as pay-
loads. This database structure supports modality-specific and
cross-modal retrieval, serving as the foundation for external
knowledge integration in the RS-RAG framework.
C. Knowledge Retrieval and Response Generation
After constructing the Multi-Modal Knowledge Vector
Database, we implement a retrieval-augmented generation
pipeline that enhances vision-language understanding by in-
corporating external knowledge retrieved via cross-modal sim-
ilarity. Given a user query q, composed of an image component
qIand a textual component qT, we first encode each input into
dense embeddings:
vT=fT(qT),vI=fI(qI), (5)
where fT(·)andfI(·)denote the CLIP-based text and im-
age encoders, respectively. To retrieve semantically relevant
knowledge, we perform similarity search in both the text and
image embedding spaces, retrieving the top- τcandidates from
each modality:
Rτ
T=Topτ(vT,Dtext),Rτ
I=Topτ(vI,Dimage),(6)
where DtextandDimage represent the text and image embedding
collections in the vector database. While the initial retrieval
from each modality yields candidates based on unimodal
similarity, these results may contain semantically redundant,
irrelevant, or inconsistent entries due to the disjoint nature of
visual and textual embedding spaces. To address this issue,
we introduce a retrieval re-ranking step that jointly considers
both modalities. Specifically, we first merge the retrieved sets
from each modality, Rfused=Rτ
T∪Rτ
I. Each candidate is then
assigned a fused similarity score via weighted combination:
score (ri) = (1 −α)·sT(ri) +α·sI(ri), (7)
where stext(ri)andsimage(ri)are the cosine similarities be-
tween the query and the candidate in the respective embedding
spaces. The weighting parameter α∈[0,1]controls the rela-
tive influence of each modality.Based on these fused scores,
we select the top- Kmost relevant candidates:
{k1, . . . , k K}=TopK({score (ri)|ri∈ R fused}).(8)
To enhance semantic coherence and eliminate redundancy
among the retrieved segments, we apply a Knowledge-
Conditioned Context Fusion module that consolidates theminto a single, contextually grounded representation. Specif-
ically, a frozen large language model Lfuseis employed to
synthesize the knowledge-conditioned context Rfrom the top-
ranked knowledge snippets:
R=Lfuse({k1, . . . , k K}). (9)
This fusion step consolidates salient content from the re-
trieved segments into a compact, context-aware representation,
thereby facilitating structured prompt construction. Given the
original user query qTand the fused knowledge context R,
we construct a retrieval-augmented prompt PqviaKnowledge-
Augmented Prompt Construction as follows:
Pq=Concat [ϕ, qT, ψ,R] (10)
where ϕis a task-specific instruction token (e.g., “Answer the
following question based on the retrieved knowledge:”), ψis
a knowledge header (e.g., “Retrieved context:”), and Ris the
fused knowledge used to support reasoning. Finally, the com-
posed prompt Pq, along with the visual input qI, is provided
to the VLM, which performs Knowledge-Grounded Response
Generation via joint multimodal reasoning and generates the
final output:
ˆy=Generate θVLM(y|Image =qI,Prompt =Pq). (11)
By leveraging retrieval from a multi-modal knowledge base,
this framework empowers the RSVLMs to go beyond purely
visual grounding by integrating both domain-specific knowl-
edge and broader world knowledge, including cultural, his-
torical, and geopolitical context. This enriched understanding
enables the model to generate more accurate, context-aware,
and semantically comprehensive outputs. In doing so, our RS-
RAG framework effectively bridges the gap between remote
sensing imagery and external knowledge sources, establishing
a retrieval-augmented generation paradigm tailored to the
unique demands of the remote sensing domain.
V. E XPERIMENTS
A. Experimental Setup
All input images are resized to 512 by 512 pix-
els before being fed into the models. We evaluate sev-
eral state-of-the-art vision-language models as baselines,
including InternVL2.5-Instruct-8B [29], Janus-Pro-7B [30],
Qwen-2.5-VL-7B-Instruct [31], and LLaMA-3.2-Vision-11B-
Instruct [32]. Building upon Qwen-2.5-VL and LLaMA-3.2-
Vision, we further develop our retrieval-augmented variants,
named RS-RAG-7B and RS-RAG-11B, respectively. To better
adapt these models to the remote sensing domain, we fine-tune
them via Low-Rank Adaptation (LoRA) [33] using instruction-
following data consisting of 1380 curated image-text pairs.
Fine-tuning is conducted for 3 epochs with a batch size of
1, using the Adamw optimizer and an initial learning rate of
1×10−4. To systematically evaluate instruction-tuned vision-
language models in the remote sensing domain, we curate a
new benchmark consisting of task-specific subsets for image
captioning (348 samples), image classification (910 samples),
and visual question answering (300 samples). All experiments

8
Question : 
Please  provide  detailed  
descriptions  for this 
remote  sensing  image .InternVL 2.5-8B:The Rainbow  Bridge , located  in Fukuoka  Prefecture,  Japan,  at coordinates  33°25′17″N 130°25′17″E, is a suspension  bridge  
spanning  the Fukuoka  Bay and connecting  the city of Fukuoka  to the island  of Hakatajima . Constructed  between  1959  and 1961 , it was the first 
suspension  bridge  in Japan  and holds  the distinction  of being  the longest  suspension  bridge  in Japan  until  1964 . The bridge’s  cultural  significance  is 
underscored  by its appearance  in the 1964  film *The  Sound  of Music*,  where  it was used  as a backdrop  for a scene…. The bridge’s  surface  albedo  
(ALBEDO)  is 0.082405882 , with  specific  values  for near  infrared  diffuse  (ALBNIRDF : 0.080038507 ), near  infrared  beam  (ALBNIRDR : 0.082505882 ), 
visible  diffuse  (ALBVISDF : 0.068899061 ), and visible  beam  (ALBVISDR : 0.069090308 ). The surface  emissivity  (EMIS)  is 0.982505359 , while  the 
evaporation  land  (EVLAND)  is 3.02e-05….
Janus -Pro-7B:The Tsukuba  Expressway , located  in Tsukuba,  Ibaraki  Prefecture,  Japan,  is a significant  infrastructure  project  that spans  10.5 kilometers  
(6.5 miles)  and was constructed  between  1969  and 1972 . Designed  to connect  Tsukuba  to T okyo,  the expressway  was the first in Japan  to be built  
using  a single -layer  concrete  slab,  which  reduced  construction  costs  and time . The expressway  features  a 10-meter -wide  median  and a 10-meter -wide  
shoulder,  with  a speed  limit  of 100 km/h  (62 mph)  and a total  of 10 lanes,  including  two lanes  for express  traffic…. Geographically,  the expressway  is 
situated  at latitude  35.7099111  and longitude  139.7100001 , with  environmental  data  indicating  a surface  albedo  (ALBEDO)  of 0.10540017 , near  
infrared  diffuse  albedo  (ALBNIRDF)  of 0.144599999 , near  infrared  beam  albedo  (ALBNIRDR)  of 0.157599999 ….
Qwen 2.5-VL-7B:The Akashi  Kaikyo  Bridge , located  in Japan,  spans  the Seto  Inland  Sea between  Awaji  Island  and Kobe,  connecting  Honshu  to Awaji  
Island . Constructed  from  1985  to 1998 , it is the longest  suspension  bridge  in the world  at 2,904 meters  (9,527 ft) with  a main  span  of 1,991 meters  
(6,532 ft). The bridge  features  two decks,  one for rail traffic  and another  for road  traffic,  with  a total  length  of 3,592 meters  (11,784 ft), including  
approach  spans . It was designed  by Taisei  Corporation  and opened  on January  22, 1998 , replacing  the Akashi  Strait  Tunnel  as the primary  link between  
Honshu  and Awaji  Island….
Llama -3.2-11B-Vision -Instruct : The Seto  Inland  Sea Bridge , located  in Japan  at coordinates  34°20′00″N 133°55′00″E, is a significant  infrastructure  
project  spanning  the Seto  Inland  Sea, connecting  the islands  of Honshu  and Shikoku . Constructed  between  1986  and 1988 , it was the longest  
suspension  bridge  in the world  at the time  of its completion,  with  a main  span  of 1,991 meters  and a total  length  of 4,936 meters . …The bridge’s  
surface  albedo  values,  including  ALBEDO  (0.114), ALBNIRDF  (0.129), ALBNIRDR  (0.144), ALBVISDF  (0.064), and ALBVISDR  (0.070), indicate  its reflective  
properties  across  different  wavelengths,  while  its surface  emissivity  (EMIS)  of 0.973 suggests  high  thermal  emission  efficiency ….
RS-RAG -11B: The remote  sensing  image  for the Seto  Ohashi  Bridge , The Seto  Ohashi  Bridge  is a series  of double -deck  bridges  connecting  Okayama  Prefecture  on Japan‘s  main  island  of 
Honshu  with  Kagawa  Prefecture  on the island  of Shikoku . Spanning  the Seto  Inland  Sea, it is one of the longest  two-tiered  bridge  systems  in the world,  with  a total  length  of 
approximately 13.1 kilometers  (8.1 miles) . …Located  at 34°23′54″N 133°48′36″E, provides  a comprehensive  analysis  of its environmental  and meteorological  parameters . The surface  
albedo  (ALBEDO)  of 0.09050411  indicates  the bridge’s  reflectivity  across  different  wavelengths,  with  specific  values  for near  infrared  diffuse  (ALBNIRDF : 0.102668583 ), near  infrared  beam  
(ALBNIRDR : 0.109455898 ), visible  diffuse  (ALBVISDF : 0.059622779 ), and visible  beam  (ALBVISDR : 0.062816069 ) albedos,  reflecting  its interaction  with  solar  radiation . The surface  emissivity  
(EMIS)  of 0.985141039  suggests  the bridge’s  ability  to emit  thermal  radiation,  while  the average  profile  soil moisture  (GWETPROF)  of 0.847956181  indicates  the surrounding  soil’s  moisture  
content . The land  cover  classifications  reveal  LC_Type 1 (IGBP  classification)  as 13, representing  urban  and built -up areas,  LC_Type 2 (UMD  classification)  also as 13, indicating  urban  and 
built -up areas,  LC_Type 3 (LAI classification)  as 10, corresponding  to sparse  vegetation,  LC_Type 4 (BGC  classification)  as 8, indicating  broadleaf  deciduous  trees,  and LC_Type 5 (Plant  
Functional  Types  classification)  as 9, representing  broadleaf  deciduous  trees .RS-RAG -7B: The Great  Seto  Bridge , located  at coordinates  34°23′54″N 133°48′36″E, is a significant  engineering  structure  spanning  the Seto  Inland  Sea and connecting  Okayama  and 
Kagawa  prefectures  in Japan . Constructed  between  1978  and 1988  as part of the Honshū–Shikoku  Bridge  Project,  the bridge  cost US$7 billion  and utilized  3.646 million  cubic  meters  of 
concrete  and 705,000 tons  of steel .…. The bridge’s  environmental  and meteorological  data  include  a surface  albedo  (ALBEDO)  of 0.09050411 , near -infrared  diffuse  albedo  (ALBNIRDF)  of 
0.102668583 , near -infrared  beam  albedo  (ALBNIRDR)  of 0.109455898 , visible  diffuse  albedo  (ALBVISDF)  of 0.059622779 , visible  beam  albedo  (ALBVISDR)  of 0.062816069 , and surface  
emissivity  (EMIS)  of 0.985141039 . Land  cover  classifications  reveal  LC_Type 1 (IGBP  classification)  as 13, indicating  urban  and built -up areas,  which  are significant  for understanding  human  
impact  on the environment ; LC_Type 2 (UMD  classification)  as 13, also representing  urban  and built -up areas .…
Ground  truth :Great  Seto  Bridge .
Fig. 4. Qualitative results of image captioning on remote sensing imagery of the Great Seto Bridge. Text in red indicates the recognized landmark name;
purple highlights retrieved world knowledge, such as historical, geographic, or cultural facts; and green denotes domain-specific knowledge, including spectral
indices, land cover, and ALBEDO values.
TABLE III
PERFORMANCE COMPARISON BETWEEN BASELINE MODELS AND OUR PROPOSED RS-RAG VARIANTS ON THE IMAGE CAPTIONING TASK USING A
SUBSET OF THE RSWK DATASET .
Model BLEU-1 BLEU-2 BLEU-3 BLEU-4 METEOR ROUGE-L CIDEr
InternVL2.5-8B [29] 0.427 0.289 0.218 0.166 0.222 0.275 0.013
Janus-Pro-7B [30] 0.370 0.245 0.183 0.139 0.197 0.246 0.007
Qwen2.5-VL-7B [31] 0.349 0.223 0.161 0.121 0.185 0.229 0.004
Llama3.2-Vision-11B [32] 0.448 0.303 0.228 0.173 0.225 0.286 0.014
RS-RAG-7B 0.409 0.263 0.193 0.144 0.206 0.240 0.065
RS-RAG-11B 0.470 0.360 0.307 0.266 0.276 0.322 0.027
are performed using 3 NVIDIA RTX A6000 GPUs, each with
48 GB of memory.
To comprehensively evaluate model performance, we adopt
a set of standard metrics tailored to each task. For image
captioning and visual question answering, we report BLEU
1, BLEU 2, BLEU 3, BLEU 4, METEOR, ROUGE L, and
CIDEr to measure the fluency, relevance, and informativeness
of generated text. For the classification task, we evaluate
both overall accuracy and per class accuracy to reflect model
performance across diverse scene categories.B. Results on Image Captioning Task
Table III summarizes the performance of baseline and
proposed models on the image captioning task over remote
sensing imagery, evaluated using a comprehensive set of
metrics including BLEU-1 to BLEU-4, METEOR, ROUGE-L,
and CIDEr. Among the baseline models, LLaMA3.2-Vision-
11B achieves the strongest performance, attaining a BLEU-
4 score of 0.173 and METEOR of 0.225, highlighting its
relative strength in generating syntactically and semantically
coherent captions. However, it still exhibits limitations in
domain-specific understanding and factual richness due to its
lack of grounding in external knowledge. In contrast, both of
our proposed retrieval-augmented models, RS-RAG-7B and

9
TABLE IV
PERFORMANCE COMPARISON BETWEEN BASELINE MODELS AND OUR PROPOSED RS-RAG VARIANTS ON THE IMAGE CLASSFICATION TASK USING A
SUBSET OF THE RSWK DATASET .
Model Overall Airport Amusement Beach Bridge Casino Church Gov. Bldg. Historic Mansion Museum Park Stadium Theater Tower University
Accuracy Park Site
InternVL2.5-8B [29] 0.329 0.959 0.250 0.667 0.385 0.000 0.000 0.152 0.158 0.023 0.000 0.672 0.790 0.039 0.053 0.654
Janus-Pro-7B [30] 0.269 0.959 0.500 0.778 0.308 0.000 0.480 0.485 0.061 0.000 0.000 0.373 0.632 0.000 0.053 0.161
Qwen2.5-VL-7B [31] 0.309 0.959 0.300 0.556 0.615 0.091 0.112 0.364 0.094 0.023 0.016 0.709 0.790 0.039 0.158 0.296
Llama3.2-Vision-11B [32] 0.340 0.959 0.300 0.667 0.462 0.091 0.020 0.061 0.052 0.047 0.278 0.746 0.816 0.039 0.053 0.568
RS-RAG-7B 0.659 0.959 0.800 0.667 0.692 0.636 0.551 0.576 0.474 0.419 0.349 0.866 0.868 0.346 0.368 0.765
RS-RAG-11B 0.842 0.980 0.900 0.944 0.769 0.727 0.820 0.849 0.655 0.512 0.714 0.843 0.842 0.769 0.526 0.803
Question : What  is the category  of this remote  sensing  image?  Answer  the 
question  using  a single  word  or phrase  from  the following  15 categories : Airport,  
Amusement  Park,  Beach,  Bridge,  Casino,  Church,  Government  Building,  
Historic  Site, Mansion,  Museum,  Park,  Stadium,  Theater,  Tower,  University .
Ground truth ： Museum.
InternVL2.5 -8B:  University    
Janus -Pro-7B:  Government Building   
Qwen2.5 -VL-7B:  Government Building   
Llama3.2 -11B:   University   
RS-RAG -11B:   Museum    
RS-RAG -7B:   Museum    
Ground truth ： Tower.
InternVL2.5 -8B:  Government Building   
Janus -Pro-7B:  Airport     
Qwen2.5 -VL-7B:  Government Building 
Llama -3.2-11B-Vision:  Historic Site    
RS-RAG -11B:   Tower     
RS-RAG -7B:   Tower     
Question : What  is the category  of this remote  sensing  image?  Answer  the 
question  using  a single  word  or phrase  from  the following  15 categories : Airport,  
Amusement  Park,  Beach,  Bridge,  Casino,  Church,  Government  Building,  
Historic  Site, Mansion,  Museum,  Park,  Stadium,  Theater,  Tower,  University .
Denver Museum of Nature and Science
Bahrain World Trade Center
Fig. 5. Qualitative results of baseline models and our RS-RAG model on
the image classification task using the RSWK dataset.
RS-RAG-11B, demonstrate consistent improvements across
all evaluation metrics. Notably, RS-RAG-11B outperforms
all baselines with a BLEU-4 of 0.266 and METEOR of
0.276, representing absolute gains of +0.093 and +0.051 over
the strongest baseline. It also achieves the highest ROUGE-
L score of 0.322, indicating improved phrase-level overlap
with human-written descriptions. While RS-RAG-7B uses a
smaller backbone, it still yields a significant performance
boost, achieving the best CIDEr score of 0.065, which suggests
enhanced alignment with human judgment of caption quality.
To better understand the effectiveness of our RS-RAG
framework, we conduct a qualitative comparison of image
captioning outputs generated by baseline models and our RS-
RAG model, as shown in Fig. 4. The selected case depicts the
Great Seto Bridge, a large-scale landmark in Japan. Baseline
models, such as InternVL2.5-8B and Qwen2.5-VL-7B, either
misidentify the landmark (e.g., predicting the Rainbow Bridge
or Akashi Kaikyo Bridge) or generate overly generic descrip-
tions without domain-grounded facts. Although LLaMA3.2-
Vision-11B produces a correct name, it lacks detailed ev-
idence from the image or environmental context. Notably,
our proposed RS-RAG-7B and RS-RAG-11B models generate
accurate, entity-aware, and knowledge-rich descriptions that
correctly identify the Great Seto Bridge and provide fac-
tual details about its coordinates, construction history, and
structural type. Furthermore, the descriptions include domain-
specific knowledge (e.g., surface albedo, emissivity, and land
cover classification), demonstrating the ability of our models to
integrate retrieved remote sensing metadata. The use of worldknowledge, such as historical background and engineering
facts, further enhances the semantic richness and correctness
of the generated captions. These results highlight the benefits
of retrieval-augmented generation in grounding VLM outputs
with both domain and general-purpose knowledge.
C. Results on Image Classification Task
Table IV reports the performance of all models on the
image classification task using a subset of the RSWK dataset,
measured by overall accuracy and per-class accuracy across 15
scene categories. Among the baselines, LLaMA-3.2-Vision-
11B achieves the highest overall accuracy at 34.0 percent, fol-
lowed closely by InternVL2.5-8B and Qwen2.5-VL-7B. De-
spite these results, baseline models perform poorly on several
knowledge-dependent categories such as Church, Mansion,
Historic Site, and Museum, indicating their limited capacity
to reason about semantically nuanced or infrequent classes.
Our RS-RAG-11B model achieves a substantial improvement,
attaining 84.2 percent overall accuracy. It outperforms all
baselines by a large margin across nearly all categories,
including challenging ones like Church with 82.0 percent
accuracy and Historic Site with 65.5 percent accuracy. RS-
RAG-7B also surpasses all baselines, reaching 65.9 percent
overall accuracy, despite using a smaller backbone. These
results demonstrate the effectiveness of integrating retrieved
world and domain knowledge, which helps the model resolve
semantic ambiguities and enhances recognition of visually
similar or context-dependent categories in remote sensing
imagery. To better understand the effectiveness of our RS-
RAG framework, we conduct a qualitative comparison of
image classification outputs generated by baseline models
and our RS-RAG variants, as shown in Fig. 5. The figure
presents two representative examples from the RSWK dataset:
the Denver Museum of Nature and Science and the Bahrain
World Trade Center. In both cases, existing vision-language
baselines misclassify the scene, assigning incorrect labels such
as ”University,” ”Government Building,” or ”Historic Site.” In
contrast, RS-RAG-7B and RS-RAG-11B correctly identify the
ground-truth categories, demonstrating a better understanding
of scene semantics. These results qualitatively highlight the
strength of retrieval-augmented generation in capturing contex-
tual cues and leveraging external knowledge to improve fine-
grained classification in complex remote sensing scenarios.
D. Results on Visual Question Answering Task
Table V presents the quantitative evaluation of various
models on the VQA task using a subset of the RSWK dataset.

10
TABLE V
PERFORMANCE COMPARISON BETWEEN BASELINE MODELS AND OUR PROPOSED RS-RAG VARIANTS ON THE IMAGE VQA TASK USING A SUBSET OF
THE RSWK DATASET
Model BLEU-1 BLEU-2 BLEU-3 BLEU-4 METEOR ROUGE-L CIDEr
InternVL2.5-8B [29] 0.170 0.106 0.073 0.051 0.130 0.178 0.204
Janus-Pro-7B [30] 0.101 0.060 0.039 0.025 0.113 0.124 0.084
Qwen2.5-VL-7B [31] 0.116 0.067 0.042 0.026 0.124 0.135 0.105
Llama3.2-Vision-11B [32] 0.092 0.056 0.037 0.024 0.114 0.115 0.064
RS-RAG-7B 0.233 0.179 0.144 0.113 0.214 0.258 0.608
RS-RAG-11B 0.250 0.197 0.161 0.128 0.249 0.266 0.536
Question: What is the seating 
capacity of this stadium?Ground truth ： The stadium has a seating capacity of 30,118. This stadium is the Chiba Lotte Marines.
Question: What major event 
took place at this remote 
sensing image in 1996?Ground truth ： In 1996, Michael Jackson performed three sold -out shows at Crown Perth's Burswood 
Dome as part of his HIStory  World Tour.InternVL2.5 -8B:The stadium has a seating capacity of 45,000 . 
Janus -Pro-7B: The seating capacity of the stadium is 50,000.
Qwen2.5 -VL-7B: The stadium has a seating capacity of 30,000 people.
Llama3.2 -11B:I‘m unable to verify the seating capacity of the stadium you are referring to, 
as you haven’t specified the name of the stadium. Could you please provide 
more information? I'll do my best to assist you.
RS-RAG -7B: 30,118
RS-RAG -11B: The seating capacity of the stadium mentioned is 30,118.   
InternVL 2.5-8B:In 1996 , the Sydney  Opera  House  hosted  the 1996  Summer  Paralympics,  which  
were  the first Paralympic  Games  to be held in a developed  country  and featured  
3,500 athletes  from  80 countries . The event  was a significant  milestone  in 
promoting  inclusivity  and athletic  excellence  among  people  with disabilities . 
Janus -Pro-7B:There  is no major  event  that took place  at this remote  sensing  image  in 1996 . 
The image  shows  a view  of the Mirage  Hotel  and Casino  in Las V egas,  Nevada,  
United  States , with no notable  events  occurring  during  that year.
Qwen 2.5-VL-7B:The 1996  Summer  Olympics,  officially  known  as the Games  of the XXVI  
Olympiad,  were  held from  July 19 to August  4, 1996 , in Atlanta,  Georgia,  
United  States . The Games  were  the first to be held in the United  States  since  the 
1984  Summer  Olympics  in Los Angeles  and the first to be held in the Southern  
Hemisphere . 
Llama 3.2-11B:I'm unable  to verify  information  about  a specific  event  at the provided  location  
in 1996 .
RS-RAG -7B: Michael  Jackson’s  sold-out performances  during  his HIStory  World  Tour. 
RS-RAG -11B: Michael  Jackson’s  sold-out performances  during  his HIStory  World  Tour in 1996 .
Fig. 6. Qualitative results of baseline models and our RS-RAG model on the
VQA task using the RSWK dataset.
Across all metrics—including BLEU, METEOR, ROUGE-
L, and CIDEr—the proposed RS-RAG variants outperform
all baselines by substantial margins. RS-RAG-11B achieves
the highest scores on BLEU-1, BLEU-4, and METEOR,
indicating more fluent and semantically relevant answers.
RS-RAG-7B also delivers competitive results, achieving the
best CIDEr score of 0.608, suggesting strong alignment with
human-annotated answers in terms of informativeness and
precision. In contrast, baseline models such as Qwen2.5-
VL-7B and Janus-Pro-7B struggle to generate accurate or
coherent responses, especially in knowledge-intensive queries.
To better understand the sources of these improvements, Fig. 6
illustrates two representative qualitative examples. In the first
case, the model is asked about the seating capacity of a
stadium. Baseline responses are either factually incorrect or
overly generic, whereas both RS-RAG models retrieve and
accurately generate the correct capacity of 30,118. In the
second example, involving a historically grounded question,
only RS-RAG models correctly identify Michael Jackson’s
concerts at Burswood Dome during his HIStory World Tour
in 1996. Other models either hallucinate unrelated content,
such as the 1996 Summer Olympics, or fail to recognize the
event entirely. These examples underscore the strength of RS-
RAG in incorporating relevant domain and world knowledge
to produce accurate, informative, and context-aware answers
in remote sensing VQA tasks.E. Ablation Studies
Effect of the number of retrieved candidates. To in-
vestigate the impact of the number of retrieved candidates
in the retrieval-augmented generation process, we conduct an
ablation study on the top- kparameter in RS-RAG, as shown
in Table VI. With the fusion weight αin Eq.7 fixed to 0.9, we
evaluate the model’s performance on the image captioning task
under different values of k. The results show that retrieving a
single, highly relevant knowledge snippet ( k= 1) yields the
best performance across all evaluation metrics. As kincreases
to 3 and 5, the performance consistently degrades, indicat-
ing that incorporating more candidates introduces semantic
redundancy or irrelevant noise. Such noise can interfere with
the model’s ability to generate concise and accurate captions.
These findings underscore the importance of retrieval precision
in the remote sensing domain and confirm that fewer, but
higher-quality, knowledge segments are more effective for
guiding the generation process.
TABLE VI
ABLATION STUDY ON THE NUMBER OF TOP -kRETRIEVED CANDIDATES
FOR RS-RAG.
Top-k BLEU-1 BLEU-2 BLEU-3 BLEU-4 METEOR ROUGE-L CIDEr
k= 1 0.562 0.449 0.385 0.336 0.308 0.362 0.072
k= 3 0.458 0.339 0.278 0.232 0.257 0.283 0.044
k= 5 0.433 0.314 0.255 0.210 0.244 0.271 0.063
Effect of fusion weight α.We further explore the effect of
the fusion weight αthat balances visual and textual similarity
in the re-ranking step (see Eq.7). TableVII reports performance
under different values of α, with the number of retrieved
candidates fixed to k= 1. The results show that α= 0.5
achieves the best overall performance, indicating that equal
weighting of visual and text similarity provides the most in-
formative guidance for downstream caption generation. When
αis too low (e.g., 0.3), text similarity dominates, leading to
less visually grounded results. Conversely, when αincreases
beyond 0.5, performance slightly degrades, suggesting that
over-reliance on visual similarity may overlook relevant textual
semantics. These findings confirm that effective cross-modal
fusion is critical for high-quality retrieval and generation.
VI. C ONCLUSION
In this work, we presented RS-RAG, a retrieval-augmented
vision-language framework designed to bridge remote sensing
imagery with structured domain and world knowledge. To

11
TABLE VII
ABLATION STUDY ON THE FUSION WEIGHT αINRS-RAG.
α BLEU-1 BLEU-2 BLEU-3 BLEU-4 METEOR ROUGE-L CIDEr
0.3 0.414 0.253 0.179 0.127 0.215 0.199 0.005
0.5 0.582 0.473 0.410 0.361 0.318 0.393 0.210
0.7 0.568 0.454 0.389 0.338 0.309 0.366 0.110
0.9 0.562 0.449 0.385 0.336 0.308 0.362 0.072
support this framework, we constructed the RSWK dataset,
a large-scale multimodal benchmark that integrates high-
resolution satellite imagery with rich textual descriptions cov-
ering over 14,000 globally recognized locations from 175
countries. By leveraging this curated knowledge base, RS-
RAG significantly improves contextual reasoning and semantic
understanding across key vision-language tasks, including
image captioning, image classification, and visual question
answering. Extensive experiments validate the effectiveness of
our approach, particularly in handling complex, knowledge-
intensive queries. We believe that both the RSWK dataset
and the RS-RAG framework provide a strong foundation
for advancing research in remote sensing vision-language
understanding and knowledge-grounded geospatial AI.
REFERENCES
[1] W. Han, C. Wen, L. Chok, Y . L. Tan, S. L. Chan, H. Zhao, and C. Feng,
“Autoencoding tree for city generation and applications,” ISPRS Journal
of Photogrammetry and Remote Sensing , vol. 208, pp. 176–189, 2024.
[2] S. Liu, S. Wang, T. Chi, C. Wen, T. Wu, and D. Wang, “An improved
combined vegetation difference index and burn scar index approach
for mapping cropland burned areas using combined data from landsat
8 multispectral and thermal infrared bands,” International Journal of
Wildland Fire , vol. 29, no. 6, pp. 499–512, 2020.
[3] C. Wen, S. Liu, X. Yao, L. Peng, X. Li, Y . Hu, and T. Chi, “A novel
spatiotemporal convolutional long short-term neural network for air
pollution prediction,” Science of the total environment , vol. 654, pp.
1091–1099, 2019.
[4] X. X. Zhu, D. Tuia, L. Mou, G.-S. Xia, L. Zhang, F. Xu, and
F. Fraundorfer, “Deep learning in remote sensing: A comprehensive
review and list of resources,” IEEE Geoscience and Remote Sensing
Magazine , vol. 5, no. 4, pp. 8–36, 2017.
[5] H. Lin, N. Li, P. Yao, K. Dong, Y . Guo, D. Hong, Y . Zhang, and C. Wen,
“Generalization-enhanced few-shot object detection in remote sensing,”
IEEE Transactions on Circuits and Systems for Video Technology , 2025.
[6] Y . Hu, J. Yuan, C. Wen, X. Lu, and X. Li, “Rsgpt: A remote sensing vi-
sion language model and benchmark,” arXiv preprint arXiv:2307.15266 ,
2023.
[7] Y . Bazi, L. Bashmal, M. M. Al Rahhal, R. Ricci, and F. Melgani, “Rs-
llava: A large vision-language model for joint captioning and question
answering in remote sensing imagery,” Remote Sensing , vol. 16, no. 9,
p. 1477, 2024.
[8] C. Pang, J. Wu, J. Li, Y . Liu, J. Sun, W. Li, X. Weng, S. Wang, L. Feng,
G.-S. Xia, and C. He, “H2rsvlm: Towards helpful and honest remote
sensing large vision language model,” 2024.
[9] W. Zhang, M. Cai, T. Zhang, Y . Zhuang, and X. Mao, “Earthgpt:
A universal multi-modal large language model for multi-sensor image
comprehension in remote sensing domain,” IEEE Transactions on Geo-
science and Remote Sensing , 2024.
[10] K. Kuckreja, M. S. Danish, M. Naseer, A. Das, S. Khan, and F. S. Khan,
“Geochat: Grounded large vision-language model for remote sensing,”
inProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , 2024, pp. 27 831–27 840.
[11] Y . Zhan, Z. Xiong, and Y . Yuan, “Skyeyegpt: Unifying remote sensing
vision-language tasks via instruction tuning with large language model,”
ISPRS Journal of Photogrammetry and Remote Sensing , vol. 221, pp.
64–77, 2025.
[12] S. Lobry, D. Marcos, J. Murray, and D. Tuia, “Rsvqa: Visual question
answering for remote sensing data,” IEEE Transactions on Geoscience
and Remote Sensing , vol. 58, no. 12, pp. 8555–8566, 2020.[13] X. Zheng, B. Wang, X. Du, and X. Lu, “Mutual attention inception net-
work for remote sensing visual question answering,” IEEE Transactions
on Geoscience and Remote Sensing , vol. 60, pp. 1–14, 2021.
[14] Y . Sun, S. Feng, X. Li, Y . Ye, J. Kang, and X. Huang, “Visual grounding
in remote sensing images,” in Proceedings of the 30th ACM International
conference on Multimedia , 2022, pp. 404–412.
[15] M. M. Al Rahhal, Y . Bazi, S. O. Alsaleh, M. Al-Razgan, M. L.
Mekhalfi, M. Al Zuair, and N. Alajlan, “Open-ended remote sensing
visual question answering with transformers,” International Journal of
Remote Sensing , vol. 43, no. 18, pp. 6809–6823, 2022.
[16] Y . Zhan, Z. Xiong, and Y . Yuan, “Rsvg: Exploring data and models
for visual grounding on remote sensing data,” IEEE Transactions on
Geoscience and Remote Sensing , vol. 61, pp. 1–13, 2023.
[17] B. Qu, X. Li, D. Tao, and X. Lu, “Deep semantic understanding of high
resolution remote sensing image,” in 2016 International conference on
computer, information and telecommunication systems (Cits) . IEEE,
2016, pp. 1–5.
[18] X. Lu, B. Wang, X. Zheng, and X. Li, “Exploring models and data
for remote sensing image caption generation,” IEEE Transactions on
Geoscience and Remote Sensing , vol. 56, no. 4, pp. 2183–2195, 2017.
[19] Z. Zhang, T. Zhao, Y . Guo, and J. Yin, “Rs5m and georsclip: A large
scale vision-language dataset and a large vision-language model for
remote sensing,” IEEE Transactions on Geoscience and Remote Sensing ,
2024.
[20] F. Liu, D. Chen, Z. Guan, X. Zhou, J. Zhu, Q. Ye, L. Fu, and J. Zhou,
“Remoteclip: A vision language foundation model for remote sensing,”
IEEE Transactions on Geoscience and Remote Sensing , 2024.
[21] H. Lin, C. Zhang, D. Hong, K. Dong, and C. Wen, “Fedrsclip: Federated
learning for remote sensing scene classification using vision-language
models,” arXiv preprint arXiv:2501.02461 , 2025.
[22] X. Li, C. Wen, Y . Hu, Z. Yuan, and X. X. Zhu, “Vision-language models
in remote sensing: Current progress and future trends,” IEEE Geoscience
and Remote Sensing Magazine , 2024.
[23] X. Li, C. Wen, Y . Hu, and N. Zhou, “Rs-clip: Zero shot remote sensing
scene classification via contrastive vision-language supervision,” Inter-
national Journal of Applied Earth Observation and Geoinformation ,
2023.
[24] K. Kuckreja, M. S. Danish, M. Naseer, A. Das, S. Khan, and F. S. Khan,
“Geochat: Grounded large vision-language model for remote sensing,”
inProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , 2024, pp. 27 831–27 840.
[25] H. Lin, D. Hong, S. Ge, C. Luo, K. Jiang, H. Jin, and C. Wen, “Rs-moe:
A vision-language model with mixture of experts for remote sensing
image captioning and visual question answering,” IEEE Transactions
on Geoscience and Remote Sensing , 2025.
[26] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark,
A. Ostrow, A. Welihinda, A. Hayes, A. Radford et al. , “Gpt-4o system
card,” arXiv preprint arXiv:2410.21276 , 2024.
[27] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng,
C. Zhang, C. Ruan et al. , “Deepseek-v3 technical report,” arXiv preprint
arXiv:2412.19437 , 2024.
[28] Q. Zhao, L. Yu, X. Li, D. Peng, Y . Zhang, and P. Gong, “Progress
and trends in the application of google earth and google earth engine,”
Remote Sensing , vol. 13, no. 18, p. 3778, 2021.
[29] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang,
X. Zhu, L. Lu et al. , “Internvl: Scaling up vision foundation models
and aligning for generic visual-linguistic tasks,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition ,
2024, pp. 24 185–24 198.
[30] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan,
“Janus-pro: Unified multimodal understanding and generation with data
and model scaling,” arXiv preprint arXiv:2501.17811 , 2025.
[31] Q. Team, “Qwen2.5-vl,” January 2025. [Online]. Available: https:
//qwenlm.github.io/blog/qwen2.5-vl/
[32] M. AI, “Llama 3.2: Revolutionizing edge ai and vision with open-
source models,” 2024. [Online]. Available: https://ai.meta.com/blog/
llama-3-2-connect-2024-vision-edge-mobile-devices/
[33] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang, L. Wang,
W. Chen et al. , “Lora: Low-rank adaptation of large language models.”
ICLR , vol. 1, no. 2, p. 3, 2022.