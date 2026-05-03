# GeoSearch: Augmenting Worldwide Geolocalization with Web-Scale Reverse Image Search and Image Matching

**Authors**: Tung-Duong Le-Duc, Hoang-Quoc Nguyen-Son, Minh-Son Dao

**Published**: 2026-04-28 09:00:59

**PDF URL**: [https://arxiv.org/pdf/2604.25390v1](https://arxiv.org/pdf/2604.25390v1)

## Abstract
Worldwide image geolocalization, which aims to predict the GPS coordinates of any image on Earth, remains challenging due to global visual diversity. Recent generative approaches based on Retrieval-Augmented Generation (RAG) and Large Multimodal Models (LMMs) leverage candidates retrieved from fixed databases for reasoning, but often struggle with scenes that are absent from the reference set. In this work, we propose GeoSearch, an open-world geolocation framework that integrates web-scale reverse image search into the RAG pipeline. GeoSearch augments LMM prompts with database-retrieved coordinates and textual evidence extracted from web pages. To mitigate noise from irrelevant content, we introduce a two-layer filtering mechanism consisting of image matching, followed by confidence-based gating. Experiments on standard benchmarks Im2GPS3k and YFCC4k demonstrate the superiority of GeoSearch under leakage-aware evaluation. Our code and data are publicly available to support reproducibility.

## Full Text


<!-- PDF content starts -->

GeoSearch: Augmenting Worldwide Geolocalization
with Web-Scale Reverse Image Search and Image Matching
Tung-Duong Le-Duc
University of Science, VNU-HCM
Ho Chi Minh City, Vietnam
23125081@student.hcmus.edu.vnHoang-Quoc Nguyen-Son∗
National Institute of Information
and Communications Technology
Tokyo, Japan
quoc-nguyen@nict.go.jpMinh-Son Dao∗
National Institute of Information
and Communications Technology
Tokyo, Japan
dao@nict.go.jp
Abstract
Worldwide image geolocalization, which aims to predict the GPS
coordinates of any image on Earth, remains challenging due to
global visual diversity. Recent generative approaches based on
Retrieval-Augmented Generation (RAG) and Large Multimodal
Models (LMMs) leverage candidates retrieved from fixed databases
for reasoning, but often struggle with scenes that are absent from
the reference set. In this work, we proposeGeoSearch, an open-
world geolocation framework that integrates web-scale reverse
image search into the RAG pipeline. GeoSearch augments LMM
prompts with database-retrieved coordinates and textual evidence
extracted from web pages. To mitigate noise from irrelevant content,
we introduce a two-layer filtering mechanism consisting of image
matching, followed by confidence-based gating. Experiments on
standard benchmarks Im2GPS3k and YFCC4k demonstrate the su-
periority of GeoSearch under leakage-aware evaluation. Our code1
and data2are publicly available to support reproducibility.
CCS Concepts
•Information systems→Information retrieval.
Keywords
Image Geolocalization, Retrieval-Augmented Generation, Large
Multimodal Models, Image Matching, Reverse Image Search
ACM Reference Format:
Tung-Duong Le-Duc, Hoang-Quoc Nguyen-Son, and Minh-Son Dao. 2026.
GeoSearch: Augmenting Worldwide Geolocalization with Web-Scale Re-
verse Image Search and Image Matching. InProceedings of the 49th Interna-
tional ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR ’26), July 20–24, 2026, Melbourne, VIC, Australia.ACM, New
York, NY, USA, 6 pages. https://doi.org/10.1145/3805712.3809936
1 Introduction
Worldwide image geolocalization aims to predict geographic coor-
dinates anywhere on Earth from visual content, enabling applica-
tions such as location-based services, navigation, and geographic
∗Corresponding authors.
1https://github.com/tungduong0708/GeoSearch
2https://huggingface.co/datasets/tduongvn/GeoSearch
ACM acknowledges that this contribution was authored or co-authored by an employee,
contractor or affiliate of a national government. As such, the Government retains a
nonexclusive, royalty-free right to publish or reproduce this article, or to allow others
to do so, for Government purposes only. Request permissions from owner/author(s).
SIGIR ’26, Melbourne, VIC, Australia
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2599-9/2026/07
https://doi.org/10.1145/3805712.3809936information retrieval. Recent generative methods based on Large
Multimodal Models (LMMs) and Retrieval-Augmented Generation
(RAG) [11, 12, 38] leverage retrieved context for reasoning, but re-
main limited by database coverage and retrieval quality, especially
for newly constructed or unseen landmarks.
To address this limitation, we proposeGeoSearch, a geolocation
framework that integrates web-scale reverse image search into the
RAG pipeline. GeoSearch builds upon G3 [ 11], which learns unified
image, text, and location representations for large-scale retrieval.
We further adopt an Earth-Centered, Earth-Fixed (ECEF) projec-
tion to improve global geographic alignment. At inference time,
GeoSearch augments LMM prompts with retrieved GPS coordinates
and textual information from reverse image search results.
Since web-scale retrieval may introduce noisy or irrelevant re-
sults, GeoSearch employs atwo-layer filtering strategyto adap-
tively select between search-augmented prediction and closed-
world baseline prediction. First, image matching is performed us-
ing SuperPoint [ 6] and LightGlue [ 17]. When matching fails, a
confidence-based gating mechanism based on the similarity be-
tween the query image and the predicted GPS coordinates is applied,
allowing the model to fall back to existing baseline methods.
Existing RAG-based approaches [ 11,12,38] build databases from
MP16 [ 14] and evaluate on Im2GPS3k [ 9] and YFCC4k [ 28]. As
these datasets are derived from Flickr3, this may cause data leak-
age. We therefore construct our database from OpenStreetView-
5M (OSV-5M) [ 2], collected from Mapillary4. Under this leakage-
aware setting, GeoSearch achieves state-of-the-art performance on
both benchmarks and remains competitive even without relying
on leaked data. This demonstrates the effectiveness of web-scale
retrieval with adaptive filtering.
Our main contributions are summarized as follows:
•We propose an adaptive open-world geolocalization frame-
work that integrates web-scale reverse image search with
RAG-based LMM reasoning and two-layer filtering.
•We adopt the ECEF projection to improve global geographic
alignment in location representation.
•We achieve state-of-the-art performance on both Im2GPS3k
and YFCC4k under leakage-aware evaluation.
2 Related Work
Worldwide Image Geolocalization.Existing methods mainly
follow three paradigms: classification, retrieval, and generative mod-
eling. (1)Classification-based methods[ 5,8,10,18,19,22,31,35]
partition the Earth into discrete geographic regions and formulate
3https://www.flickr.com/
4https://www.mapillary.com/arXiv:2604.25390v1  [cs.IR]  28 Apr 2026

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Tung-Duong Le-Duc, Hoang-Quoc Nguyen-Son, and Minh-Son Dao
Query ImageSimilarity
Score
Reverse Image
SearchSearch
resultsPrompt
Augmentation
Location
GeocodingRetrieval
DatabaseVision
Encoder
Retrieval
Database
Query Image
GPS
CandidatesLocation
EncoderVision
EncoderQuery Image Link Image
Layer  1
Image
Matching
Layer  2
Similarity Score σ
GeoSearch BaselineVision
EncoderLocation
Encoder
Text
EncoderProjection
A photo taken in
Paris, France
48.85 2.35Z
XY
Frozen
Trainable
Two-Layer  Filtering
System
Final Predicted GPSPredicted
GPSPredicted
GPS(A) Pr eprocessing Phase
(B) Infer ence PhaseTwo-Layer  Filtering System
Location Infer ence Location Refinement
 not
samesame
σ ≥ α σ < αFarthest
imagesNearest 
images
Location: ...
Your task is to determine the geographic coordinates (latitude and longitude) of the specified location by
following these steps:
1. Attempt to find the exact GPS coordinates using reliable online sources such as maps or satellite
imagery .
2. If the exact location is not available, find the coordinates of a nearby or adjacent place (e.g., a
recognizable landmark, building, road, or intersection).
3. If no specific nearby location can be found, use the coordinates of the broader area (e.g., the center of
the city).
Return your answer in the following JSON format: {"latitude": float, "longitude": float }
Location Pr ediction Pr ompt Location Geocoding Pr omptYou are an expert in geo-localization. Analyze the image and determine the most precise possible location-
ideally identifying the exact building, landmark, or facility , not just the city . Return your final answer as a
location name.
Coordinates:  these are coordinates of some similar images (    ), and these are coordinates of some dissimilar
images (    ).
Links: ...
Respond with only the following JSON structure (no extra text, markdown, or comments):  {"location": string }
Guidelines: Analyze all available information including images and provided links. If you can't identify a
specific building, provide the city name.  The location field should be a descriptive name of the place (e.g.,
"Eiffel Tower , Paris, France" or "T imes Square, New York, USA").
Figure 1: Overview of the GeoSearch framework.
localization as a multi-class prediction problem, but suffer from
spatial quantization. (2)Retrieval-based methodsmatch queries
against geotagged images [ 13,23,25,39] or GPS galleries [ 30].
Visual retrieval, as adopted in visual place recognition (VPR), de-
pends heavily on database coverage and diversity [ 3,4,26,29,34],
and degrades when similar locations are missing. (3)Generative
methodsinclude diffusion-based models [ 7,33] and LMM-based
approaches [ 15,16,24,36]. Recent RAG-based methods [ 11,12,38]
integrate retrieved references for localization, but operate in closed-
world settings with static databases, limiting generalization to un-
seen landmarks.
Web-Scale Retrieval and Open-World Image Geolocation.
Web-scale visual retrieval has been widely studied in landmark
recognition and image indexing [ 32,37] and is now deployed in
reverse image search systems for Open-Source Intelligence and
digital forensics [ 1,20]. GeoSearch extends web-scale retrieval by
incorporating automated reasoning for open-world geolocation.
3 Proposed Approach
As shown in Figure 1, GeoSearch has two phases:(A) Prepro-
cessing, which learns unified image-text-location embeddings and
builds a retrieval database; and(B) Inference, which combines data-
base retrieval and reverse image search to generate GPS candidates,
then applies similarity ranking and filtering for final prediction.
3.1 Preprocessing Phase
3.1.1 Model Architecture.Our model consists of three encoders
for image, text, and location modalities.
Image Encoder.We adopt the pre-trained CLIP [ 21] ViT-L/14
as a shared visual backbone 𝑉. On top of this backbone, two par-
allel trainable transformation heads are applied to align imagefeatures with different modalities: given an input image 𝐼, the
visual representation 𝑉(𝐼) is mapped toe𝑖𝑚𝑔
𝑡𝑥𝑡=𝑓𝑡𝑥𝑡(𝑉(𝐼)) and
e𝑖𝑚𝑔
𝑙𝑜𝑐=𝑓𝑙𝑜𝑐(𝑉(𝐼)) , where𝑓𝑡𝑥𝑡and𝑓𝑙𝑜𝑐are feed-forward projection
layers for text and location alignment, respectively.
Text Encoder.For the text modality, a location description 𝐷𝑙𝑜𝑐is
encoded by the CLIP text encoder 𝑇and further projected by a train-
able head𝑔 𝑡𝑥𝑡to obtain the text embeddinge 𝑡𝑥𝑡=𝑔𝑡𝑥𝑡(𝑇(𝐷𝑙𝑜𝑐)).
Location Encoder.To eliminate the geometric distortions inher-
ent in 2D map projections used in prior work (e.g., Mercator [ 11],
Equal Earth Projection (EEP) [ 30]), we represent GPS coordinates
using theEarth-Centered, Earth-Fixed (ECEF)system to pre-
serve global spatial continuity. Given a coordinate with latitude 𝜙,
longitude𝜆, and altitude ℎ(set to 0), we map it to a continuous 3D
Cartesian vectorp=[𝑥,𝑦,𝑧]𝑇:
𝑥=(𝑁(𝜙)+ℎ)cos𝜙cos𝜆
𝑦=(𝑁(𝜙)+ℎ)cos𝜙sin𝜆
𝑧=((1−𝜀2)𝑁(𝜙)+ℎ)sin𝜙
where𝑁(𝜙) is the prime vertical radius of curvature and 𝜀is the
Earth’s eccentricity.
To capture multi-scale geographic semantics, we adapt hier-
archical Random Fourier Features [ 27] to 3D ECEF coordinates.
Specifically, given an ECEF vectorp, we apply a sinusoidal map-
ping𝛾(p,𝜎𝑖)=[cos( 2𝜋Wp),sin( 2𝜋Wp)]𝑇, whereW∼N( 0,𝜎𝑖)
and𝜎𝑖controls the frequency scale of the𝑖-th layer. The resulting
features are processed by scale-specific multi-layer perceptrons
ℎ𝑖and aggregated across 𝐾layers to produce the final location
embeddinge 𝑙𝑜𝑐=Í𝐾
𝑖=1ℎ𝑖(𝛾(p,𝜎𝑖)).

GeoSearch: Augmenting Worldwide Geolocalization with Web-Scale Reverse Image Search and Image Matching SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
3.1.2 Training.Following prior work [ 11,30], we use a contrastive
objective based on Information Noise-Contrastive Estimation (In-
foNCE) to align image, text, and GPS representations. Let 𝑎and𝑏
denote two modalities with ℓ2-normalized embeddingse𝑎ande𝑏.
The loss from𝑎to𝑏is defined as
L𝑎,𝑏=−1
|B||B|∑︁
𝑖=1logexp(⟨e𝑎
𝑖,e𝑏
𝑖⟩/𝛽)
Í|B|
𝑗=1exp(⟨e𝑎
𝑖,e𝑏
𝑗⟩/𝛽)(1)
Here,⟨·,·⟩ denotes cosine similarity, 𝛽is the temperature set
according to [ 11], and|B|is the batch size. We adopt a symmetric
formulation and define the overall objective as L=1
2(Limg,txt+
Ltxt,img+L img,loc+L loc,img).
3.1.3 Database Construction.The retrieval database is built by
encoding each reference image into a unified multi-modal vector.
For an image 𝐼, the stored representation concatenates raw visual
features with text-aligned and location-aligned embeddings:v 𝑑𝑏=
Concat
𝑉(𝐼),e𝑖𝑚𝑔
𝑡𝑥𝑡,e𝑖𝑚𝑔
𝑙𝑜𝑐
.
3.2 Inference Phase
3.2.1 Location Inference.Given a query image 𝐼𝑞, we first collect
closed-world and open-world evidence.
•Closed-World Retrieval.Top- 𝑘nearest and farthest neigh-
bors of𝐼𝑞are retrieved from the database based on visual
similarity, yielding a set of their GPS coordinatesG 𝑑𝑏.
•Open-World Retrieval.We perform reverse image search
via Google Lens and extract textual content from the re-
trieved web pages, denoted asT 𝑤𝑒𝑏.
Prompt Construction and Generation.Following G3 [ 11], to ob-
tain diverse GPS candidates for the Location Refinement stage, we
construct𝑛prompts (𝑛=4), each incorporating a different number
of GPS coordinates to encourage prediction diversity. We further
extend this design by incorporating content from the top- 𝑚web
contexts inT𝑤𝑒𝑏(𝑚= 5) into each prompt, providing richer contex-
tual evidence. The resulting prompts are fed intoGemini 2.0 Flash
to generate location descriptions, which are later converted into
GPS coordinates. This two-stage generation process reduces quan-
tization errors compared to directly predicting coordinates.
Location Geocoding.The generated locations are converted into
GPS coordinates using OpenStreetMap Nominatim5, with Gemini
as a fallback. This forms a set of GPS candidatesC 𝑔𝑝𝑠.
3.2.2 Location Refinement.To determine the optimal candidate in
C𝑔𝑝𝑠, each candidate 𝑐∈C𝑔𝑝𝑠is embedded ase 𝑙𝑜𝑐(𝑐)and matched
with the query image 𝐼𝑞using cosine similarity. The candidate with
the highest score 𝜎is selected as the search-augmented prediction
𝑃𝑠𝑒𝑎𝑟𝑐ℎ and passed to filtering.
3.2.3 Two-Layer Filtering System.To handle noisy web results, we
employ a filtering mechanism to select between 𝑃𝑠𝑒𝑎𝑟𝑐ℎ and the
baseline prediction 𝑃𝑏𝑎𝑠𝑒from existing methods, which relies only
on closed-world database coordinates.
Layer 1: Image Matching.We verify consistency between 𝐼𝑞and
the retrieved web image 𝐼𝑙𝑖𝑛𝑘by extracting keypoints withSuper-
Point[ 6], matching them usingLightGlue[ 17]. A homography is
then estimated using RANSAC, with a reprojection error threshold
of𝜏𝑟pixels (𝜏𝑟=4.0). Let𝑀be the number of matched keypoints
5https://nominatim.org/and𝜌be the inlier ratio after RANSAC. The image pair is consid-
ered to depict the same scene if 𝑀≥𝜏𝑚and𝜌≥𝜏𝑖𝑛(𝜏𝑚=50and
𝜏𝑖𝑛=0.5). If satisfied, 𝑃𝑠𝑒𝑎𝑟𝑐ℎ is accepted as final output; otherwise,
the system proceeds to Layer 2. All filtering hyperparameters (e.g.,
𝜏𝑟,𝜏𝑚,𝜏𝑖𝑛) are tuned on MP16-Search6(a validation set from MP16-
Pro) via iterative evaluation and kept constant across experiments.
Layer 2: Confidence Gating.When matching fails or 𝐼𝑙𝑖𝑛𝑘is un-
available, we rely on the model confidence by thresholding the
similarity score 𝜎of𝑃𝑠𝑒𝑎𝑟𝑐ℎ . If𝜎≥𝛼 (𝛼=0.21),𝑃𝑠𝑒𝑎𝑟𝑐ℎ is kept; oth-
erwise, the system returns 𝑃𝑏𝑎𝑠𝑒. This guarantees that GeoSearch
does not underperform the closed-world baseline in noisy scenarios.
4 Experiments
4.1 Experimental Setup
Dataset and Evaluation Metrics.As discussed in Section 1, exist-
ing datasets such as MP16 [ 14], Im2GPS3k [ 9], and YFCC4k [ 28]
are all derived from Flickr, raising concerns about data leakage.
To address this issue, we build our retrieval database from the
OSV-5M training set [ 2], which is sourced from Mapillary. For de-
velopment, we split MP16-Pro [ 11] into a 3,000-image validation
set (MP16-Search) for hyperparameter tuning and a training set
with the remaining images. Localization accuracy is reported based
on geodesic distance at thresholds of 1, 25, 200, 750, and 2500 km.
Training and Inference.We train the encoders on the MP16-
Pro [ 11] training split for 2 epochs using AdamW (learning rate
3×10−5, weight decay1×10−6) with a batch size of 256. The temper-
ature𝛽in Equation 1 is set to 3.99. All experiments are conducted
on a single NVIDIA H100 GPU. During inference, we use FAISS
for nearest-neighbor retrieval andGemini Flash 2.0for reasoning
with temperature 1.0.
4.2 Comparison with SOTA Methods
As shown in Table 1, we compare GeoSearch with recent RAG-based
methods, including Img2Loc [ 38], G3 [ 11], and GeoRanker [ 12],
on Im2GPS3k and YFCC4k using the OSV-5M database. Under
this leakage-aware setting, GeoSearch consistently achieves the
best performance, particularly at fine-grained levels. On Im2GPS3k,
GeoSearch with the GeoRanker baseline reaches 23.56% at 1 km
and 89.59% at 2500 km, substantially surpassing all baselines. On
YFCC4k, GeoSearch with the G3 baseline achieves the highest per-
formance, reaching 17.53% at 1 km and 79.85% at 2500 km.
Furthermore, under the data leakage setting in Table 2, although
existing methods benefit from MP16-based retrieval, GeoSearch
with the GeoRanker baseline still outperforms across all thresholds
on Im2GPS3k, further validating its robustness.
4.3 Ablation Study
We conduct an ablation study on Im2GPS3k with GeoRanker base-
line to analyze the effect of each component, as shown in Table 3.
w/o ClosedWorld:This variant removes the closed-world re-
trieval component, relying only on reverse image search. Perfor-
mance degrades across all distance thresholds, with larger drops at
coarse levels. This highlights the importance of internal database
candidates, which provide complementary geographic coverage
when web retrieval alone is insufficient or noisy.
6https://huggingface.co/datasets/tduongvn/MP16-Search

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Tung-Duong Le-Duc, Hoang-Quoc Nguyen-Son, and Minh-Son Dao
MethodStreet
1 kmCity
25 kmRegion
200 kmCountry
750 kmContinent
2500 km
Im2GPS3k[9]
[L]kNN,𝜎=4[31] 7.20 19.40 26.90 38.90 55.90
PlaNet [35] 8.50 24.80 34.30 48.40 64.60
CPlaNet [22] 10.20 26.50 34.60 48.60 64.60
ISNs [18] 10.50 28.00 36.60 49.70 66.00
Translocator [19] 11.80 31.10 46.70 58.90 80.10
GeoDecoder [5] 12.80 33.50 45.90 61.00 76.10
GeoCLIP [30] 14.11 34.47 50.65 69.67 83.82
PIGEON [8] 11.30 36.70 53.80 72.40 85.30
Img2Loc†[38] 12.89 37.96 52.75 67.61 79.17
G3†[11] 13.51 36.27 50.45 67.03 81.92
GeoRanker†[12] 10.08 36.50 54.79 72.87 86.65
GLOBE [16] 9.84 40.18 56.19 71.45 82.38
GeoBayes [24] 6.30 34.70 53.60 73.70 85.90
Geo-R [36] 18.10 41.53 58.31 75.33 86.42
GeoSearch (Ours)
+ Img2Loc†23.5655.02 66.97 79.75 89.32
+ G3†23.49 54.92 66.87 79.5889.42
+ GeoRanker†23.56 55.06 67.10 79.81 89.59
YFCC4k[28]
[L]kNN,𝜎=4[31] 2.30 5.70 11.00 23.50 42.00
PlaNet [35] 5.60 14.30 22.20 36.40 55.80
CPlaNet [22] 7.90 14.80 21.90 36.40 55.50
ISNs [18] 6.50 16.20 23.80 37.40 55.00
Translocator [19] 8.40 18.60 27.00 41.10 60.40
GeoDecoder [5] 10.30 24.40 33.90 50.00 68.70
GeoCLIP [30] 9.59 19.31 32.63 55.00 74.69
PIGEON [8] 10.40 23.70 40.60 62.20 77.70
Img2Loc†[38] 5.72 17.79 30.71 44.76 54.53
G3†[11] 6.55 18.14 31.70 49.80 68.63
GeoRanker†[12] 5.20 18.21 33.95 55.49 73.41
GeoBayes [24] 4.90 16.10 30.90 55.80 75.40
Geo-R [36] 10.47 22.67 40.04 60.83 75.84
GeoSearch (Ours)
+ Img2Loc†17.4835.2148.1563.36 79.19
+ G3†17.53 35.21 48.19 63.49 79.85
+ GeoRanker†17.50 35.1948.1963.36 79.28
Table 1: Main results on Im2GPS3k and YFCC4k. “+” denotes
integration of GeoSearch with baseline models at the filtering
stage to select between GeoSearch and baseline output.†
denotes models using OSV-5M for retrieval. Best results are
in bold, and second-best are underlined.
MethodStreet
1 kmCity
25 kmRegion
200 kmCountry
750 kmContinent
2500 km
Img2Loc 17.1045.14 57.87 72.91 84.68
G3 16.65 40.94 55.56 71.24 84.68
GeoRanker 18.79 45.0561.49 76.31 89.29
GeoSearch 23.56 55.06 67.10 79.81 89.59
Table 2: Results on Im2GPS3k under data leakage.
w/o Location Geocoding:Instead of generating a location de-
scription followed by geocoding, the LMM directly outputs GPS
coordinates. While this slightly improves coarse-level accuracy,
it causes a noticeable drop at street level (2.77%). This suggests
that the two-stage prediction process reduces coordinate noise and
improves fine-grained localization.
w/o Layer 1 (Image Matching):In this variant, predicted GPS
coordinates are passed directly to Layer 2. Similar to w/o Location
Geocoding, this results in a slight decrease at fine-grained levels
and a small increase at coarser levels. This indicates that Image
Matching helps maintain candidate quality by filtering visually
inconsistent matches early.
w/o Layer 2 (Confidence Gating):This variant disables the
confidence-based filtering stage, causing the system to fall back toMethodStreet
1 kmCity
25 kmRegion
200 kmCountry
750 kmContinent
2500 km
GeoSearch 23.5655.06 67.10 79.81 89.59
w/o ClosedWorld 22.46 54.15 64.60 75.44 86.09
w/o Geocoding 20.79 54.3268.37 81.08 90.59
w/o ImgMatch 23.52 55.0967.13 79.88 89.69
w/o ConfGate 19.85 48.25 62.60 76.38 88.25
Table 3: Ablation study on Im2GPS3k.
MethodStreet
1 kmCity
25 kmRegion
200 kmCountry
750 kmContinent
2500 km
Img2Loc (MP16) 17.10 45.14 57.87 72.91 84.68
Img2Loc (OSV-5M) 12.89 37.96 52.75 67.61 79.17
G3 (MP16) 16.65 40.94 55.56 71.24 84.68
G3 (OSV-5M) 13.51 36.27 50.45 67.03 81.92
GeoRanker (MP16) 18.79 45.05 61.49 76.31 89.29
GeoRanker (OSV-5M) 10.08 36.50 54.79 72.87 86.65
GeoSearch (MP16) 23.92 55.49 67.77 80.28 90.19
GeoSearch (OSV-5M) 23.56 55.06 67.10 79.81 89.59
Table 4: Comparison of retrieval databases on Im2GPS3k.
MethodStreet
1 kmCity
25 kmRegion
200 kmCountry
750 kmContinent
2500 km
GeoSearch (LLaVA) 17.08 42.08 54.05 67.23 79.01
GeoSearch (GPT) 21.72 50.62 61.26 72.14 83.98
GeoSearch (Gemini) 23.56 55.06 67.10 79.81 89.59
Table 5: Impact of LMMs on GeoSearch (Im2GPS3k).
Figure 2: Geographic distributions of GPS galleries.
the baseline when candidates fail Layer 1. Its removal leads to sub-
stantial degradation across all distance thresholds, demonstrating
its central role in filtering unreliable predictions and maintaining
stable localization performance.
4.4 Effect of Retrieval Database
Table 4 shows that most retrieval-based methods suffer noticeable
performance drops on Im2GPS3k when using the OSV-5M database
compared to MP16, revealing the impact of data leakage. Meanwhile,
GeoSearch with the GeoRanker baseline remains competitive, with
only a slight decrease under this leakage-aware setting.
4.5 Effect of LMMs
We analyze the impact of different LMMs in GeoSearch with the
GeoRanker baseline. As shown in Table 5, Gemini (Gemini Flash 2.0)
performs best across all thresholds, followed by GPT (GPT-4o-mini)
and LLaVA (LLaVA-Next-LLaMA3-8B).
4.6 Effectiveness of ECEF Projection
Following [ 30], we evaluate our ECEF-based encoder via image-
to-GPS retrieval using cosine similarity, enabling comparison with
GeoCLIP (EEP) and G3 (Mercator). Since GeoCLIP uses a 100K-
point gallery with a distribution similar to Im2GPS3k, which may
cause data leakage (Figure 2), we instead construct a uniformly
distributed global gallery with 500K points for fair evaluation.
As shown in Table 6, GeoSearch outperforms both baselines
across all thresholds, with notable gains in coarse localization
(+8.98% at 750 km), validating the ECEF projection’s effectiveness.
4.7 Hyperparameter Analysis
We analyze the hyperparameters of Layer 1 and Layer 2 under a
unified binary classification between 𝑃𝑠𝑒𝑎𝑟𝑐ℎ and𝑃𝑏𝑎𝑠𝑒on MP16-
Search. Layer 1 thresholds are tuned on all queries and the failed

GeoSearch: Augmenting Worldwide Geolocalization with Web-Scale Reverse Image Search and Image Matching SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
MethodStreet
1 kmCity
25 kmRegion
200 kmCountry
750 kmContinent
2500 km
G3 Model 0.17 5.24 25.89 47.91 69.80
GeoCLIP 0.207.17 29.10 52.82 74.11
GeoSearch Model 0.20 9.34 31.13 56.89 74.94
Table 6: Performance on Im2GPS3k with GPS Gallery.
0 250 500 750 1000
Total Matches (τm)0.000.010.02DensityBaseline Preferred
Search Preferred
0.00 0.25 0.50 0.75 1.00
Inlier Ratio (τin)01020DensityBaseline Preferred
Search Preferred
0.15 0.18 0.21 0.24 0.27
Similarity Threshold (α)0.620.64Accuracy
0.15 0.18 0.21 0.24 0.27
Similarity Threshold (α)0.730.750.78F1 Score
Figure 3: Filtering hyperparameter analysis on MP16-Search.
Method Img2Loc G3 GeoRanker GeoSearch
Time (s) 2.48 2.70 4.48 5.96
Table 7: Average inference time (seconds).
cases are used to optimize Layer 2. Here, 𝑃𝑠𝑒𝑎𝑟𝑐ℎ is obtained from the
full pipeline with reverse image search, while 𝑃𝑏𝑎𝑠𝑒relies on closed-
world GPS candidates only. A case is labeled asSearch Preferredif
𝑃𝑠𝑒𝑎𝑟𝑐ℎ outperforms 𝑃𝑏𝑎𝑠𝑒in distance thresholds or geodesic error;
otherwise, it is labeled asBaseline Preferred.
As shown in Figure 3,𝜏 𝑚=50and𝜏 𝑖𝑛=0.5separate unreliable
matches from reliable ones, where 𝑃𝑠𝑒𝑎𝑟𝑐ℎ is more likely to outper-
form𝑃𝑏𝑎𝑠𝑒. On the remaining queries after Layer 1, 𝛼=0.21yields
the highest accuracy and F1-score over 𝛼∈[ 0,1], indicating reliable
use of web evidence. These values are used in all experiments.
4.8 Efficiency Analysis
Inference Time.As shown in Table 7, GeoSearch has higher la-
tency than other RAG-based methods due to longer web-augmented
prompts and the additional geocoding step for GPS extraction.
Token Consumption.For location generation, the token usage is
estimated as462+19𝑛𝑐+2070𝑛𝑙, where 462 is the fixed prompt
and image cost, and 𝑛𝑐and𝑛𝑙denote the numbers of reference
coordinates and retrieved web pages, respectively. Each web content
is truncated to the first 2000 characters, and the average response
is about 25 tokens. Fallback geocoding into GPS coordinates costs
approximately 170 tokens with an average response of 35 tokens.
4.9 Failure Cases Analysis
Despite strong performance, several challenging failure modes re-
main. Figure 4 presents representative examples from Im2GPS3k
with large localization errors, grouped into two categories.
Case 1: Matching link images exist but fail to localize cor-
rectly.As shown in Figure 4(a), visually similar images can be
retrieved from the Internet, but they do not provide sufficient struc-
tural or geographic cues to localize. Failures also occur when similar
structures appear across multiple locations, leading to noisy and
ambiguous web search results. Additionally, even with reliable
matches, GeoSearch may still have low confidence in its predic-
tion and fall back to the baseline output, resulting in incorrect
predictions.
Query
ImageLink
Image
(a)(b)
(c)Figure 4: Failure cases in Im2GPS3k (geodesic distance >
2500 km from ground truth). (a) Query image with matched
link image. (b–c) Query images without reliable matches,
including human-centered scenes or generic content (e.g.,
natural scenery or everyday objects).
Case 2: No reliable matching link images exist.In these
cases, the query images lack distinctive location-specific content,
making it difficult to retrieve useful matches from the Internet.
Figure 4(b) shows human-centered scenes dominated by ordinary
people, where the background provides little contextual or geo-
graphic information. Figure 4(c) presents natural or generic scenes,
including vegetation, mountains, and miscellaneous objects such as
food or vehicles, which are not specific enough to indicate a unique
location. As a result, these images provide weak geographic signals
and are difficult to localize using image retrieval alone.
4.10 Limitations
GeoSearch relies on web-scale image search for visual and textual
context, which may raise privacy and ethical concerns. Moreover,
extended prompts and external geocoding increase inference time
compared to existing RAG-based methods. Future work will explore
more efficient prompt construction and geocoding, together with
privacy-aware retrieval strategies, to mitigate these limitations.
5 Conclusion
We proposed GeoSearch, an open-world image geolocalization
framework that integrates web-scale reverse image search with
retrieval-augmented multimodal reasoning. By leveraging external
evidence, ECEF-based location encoding, and a two-layer filter-
ing strategy, GeoSearch achieves state-of-the-art performance on
Im2GPS3k and YFCC4k under leakage-aware evaluation. These
results demonstrate its effectiveness and robustness on standard
benchmarks. Further improvements will focus on efficiency and
privacy while maintaining high accuracy at scale.
Acknowledgments
This research is supported by research funding from Faculty of
Information Technology, University of Science, Vietnam National
University - Ho Chi Minh City.

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Tung-Duong Le-Duc, Hoang-Quoc Nguyen-Son, and Minh-Son Dao
References
[1]Sahar Abdelnabi, Rakibul Hasan, and Mario Fritz. 2022. Open-domain, content-
based, multi-modal fact-checking of out-of-context images via online resources. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR). 14940–14949.
[2]Guillaume Astruc, Nicolas Dufour, Ioannis Siglidis, Constantin Aronssohn, Nacim
Bouia, Stephanie Fu, Romain Loiseau, Van Nguyen Nguyen, Charles Raude,
Elliot Vincent, et al .2024. Openstreetview-5m: The many roads to global visual
geolocation. InProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR). 21967–21977.
[3]Giovanni Barbarani, Mohamad Mostafa, Hajali Bayramov, Gabriele Trivigno,
Gabriele Berton, Carlo Masone, and Barbara Caputo. 2023. Are local features
all you need for cross-domain visual place recognition?. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 6155–
6165.
[4]Gabriele Berton, Carlo Masone, and Barbara Caputo. 2022. Rethinking visual geo-
localization for large-scale applications. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR). 4878–4888.
[5]Brandon Clark, Alec Kerrigan, Parth Parag Kulkarni, Vicente Vivanco Cepeda,
and Mubarak Shah. 2023. Where we are and what we’re looking at: Query based
worldwide image geo-localization using hierarchies and scenes. InProceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
23182–23190.
[6]Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. 2018. Superpoint:
Self-supervised interest point detection and description. InProceedings of the
IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 224–236.
[7]Nicolas Dufour, Vicky Kalogeiton, David Picard, and Loic Landrieu. 2025. Around
the world in 80 timesteps: A generative approach to global visual geolocation. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR). 23016–23026.
[8]Lukas Haas, Michal Skreta, Silas Alberti, and Chelsea Finn. 2024. Pigeon: Predict-
ing image geolocations. InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR). 12893–12902.
[9]James Hays and Alexei A Efros. 2008. Im2gps: estimating geographic information
from a single image. InProceedings of the IEEE Conference on Computer Vision
and Pattern Recognition (CVPR). IEEE, 1–8.
[10] Mike Izbicki, Evangelos E Papalexakis, and Vassilis J Tsotras. 2019. Exploiting the
earth’s spherical geometry to geolocate images. InJoint European Conference on
Machine Learning and Knowledge Discovery in Databases (ECML PKDD). Springer,
3–19.
[11] Pengyue Jia, Yiding Liu, Xiaopeng Li, Xiangyu Zhao, Yuhao Wang, Yantong
Du, Xiao Han, Xuetao Wei, Shuaiqiang Wang, and Dawei Yin. 2024. G3: an
effective and adaptive framework for worldwide geolocalization using large multi-
modality models.Advances in Neural Information Processing Systems (NeurIPS)
37, 53198–53221.
[12] Pengyue Jia, Seongheon Park, Song Gao, Xiangyu Zhao, and Sharon Li. 2025.
GeoRanker: Distance-Aware Ranking for Worldwide Image Geolocalization.Ad-
vances in Neural Information Processing Systems (NeurIPS).
[13] Ahmad Khaliq, Michael Milford, and Sourav Garg. 2022. Multires-netvlad: Aug-
menting place recognition training with low-resolution imagery.IEEE Robotics
and Automation Letters (RA-L)7, 2, 3882–3889.
[14] Martha Larson, Mohammad Soleymani, Guillaume Gravier, Bogdan Ionescu, and
Gareth JF Jones. 2017. The benchmarking initiative for multimedia evaluation:
MediaEval 2016.IEEE MultiMedia24, 1, 93–96.
[15] Ling Li, Yu Ye, Yao Zhou, Bingchuan Jiang, and Wei Zeng. 2024. Georeasoner:
Geo-localization with reasoning in street views using a large vision-language
model.International Conference on Machine Learning (ICML).
[16] Ling Li, Yao Zhou, Yuxuan Liang, Fugee Tsung, and Jiaheng Wei. 2025. Recog-
nition through reasoning: Reinforcing image geo-localization with large vision-
language models.Advances in Neural Information Processing Systems (NeurIPS)
(2025).
[17] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. 2023. Lightglue:
Local feature matching at light speed. InProceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV). 17627–17638.
[18] Eric Muller-Budack, Kader Pustu-Iren, and Ralph Ewerth. 2018. Geolocation
estimation of photos using a hierarchical model and scene classification. In
Proceedings of the European Conference on Computer Vision (ECCV). 563–579.
[19] Shraman Pramanick, Ewa M Nowara, Joshua Gleason, Carlos D Castillo, and
Rama Chellappa. 2022. Where in the world is this image? transformer-based geo-
localization in the wild. InProceedings of the European Conference on Computer
Vision (ECCV). Springer, 196–215.
[20] Peng Qi, Zehong Yan, Wynne Hsu, and Mong Li Lee. 2024. Sniffer: Multimodal
large language model for explainable out-of-context misinformation detection. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR). 13052–13062.
[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,et al.2021. Learning transferable visual models from natural language supervision.
InInternational Conference on Machine Learning (ICML). PmLR, 8748–8763.
[22] Paul Hongsuck Seo, Tobias Weyand, Jack Sim, and Bohyung Han. 2018. Cplanet:
Enhancing image geolocalization by combinatorial partitioning of maps. InPro-
ceedings of the European Conference on Computer Vision (ECCV). 536–551.
[23] Davide Sferrazza, Gabriele Berton, Gabriele Trivigno, and Carlo Masone. 2025.
To Match or Not to Match: Revisiting Image Matching for Reliable Visual Place
Recognition. InProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR). 2849–2860.
[24] Weimin Shi, Xiang Li, Kaige Li, Junhao Fang, Qiang Zhou, Qichuan Geng, and
Zhong Zhou. 2026. GeoBayes: Probabilistic Image Geo-Localization Inference via
Sequential Bayesian Updating.Proceedings of the AAAI Conference on Artificial
Intelligence (AAAI)(Mar. 2026), 8997–9005.
[25] Yujiao Shi, Xin Yu, Dylan Campbell, and Hongdong Li. 2020. Where am i looking
at? joint location and orientation estimation by cross-view matching. InPro-
ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR). 4064–4072.
[26] Xun Sun, Yuanfan Xie, Pei Luo, and Liang Wang. 2017. A dataset for benchmark-
ing image-based localization. InProceedings of the IEEE Conference on Computer
Vision and Pattern Recognition (CVPR). 7436–7444.
[27] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin
Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng.
2020. Fourier features let networks learn high frequency functions in low dimen-
sional domains.Advances in Neural Information Processing Systems (NeurIPS)33,
7537–7547.
[28] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni,
Douglas Poland, Damian Borth, and Li-Jia Li. 2016. Yfcc100m: The new data in
multimedia research.Communications of the ACM (CACM)59, 2, 64–73.
[29] Akihiko Torii, Relja Arandjelovic, Josef Sivic, Masatoshi Okutomi, and Tomas
Pajdla. 2015. 24/7 place recognition by view synthesis. InProceedings of the IEEE
Conference on Computer Vision and Pattern Recognition (CVPR). 1808–1817.
[30] Vicente Vivanco Cepeda, Gaurav Kumar Nayak, and Mubarak Shah. 2023. Geoclip:
Clip-inspired alignment between locations and images for effective worldwide
geo-localization.Advances in Neural Information Processing Systems (NeurIPS)36,
8690–8701.
[31] Nam Vo, Nathan Jacobs, and James Hays. 2017. Revisiting im2gps in the deep
learning era. InProceedings of the IEEE International Conference on Computer
Vision (ICCV). 2621–2630.
[32] Xin-Jing Wang, Zheng Xu, Lei Zhang, Ce Liu, and Yong Rui. 2012. Towards
indexing representative images on the web. InProceedings of the 20th ACM
International Conference on Multimedia (ACM-MM). 1229–1238.
[33] Zhangyu Wang, Jielu Zhang, Zhongliang Zhou, Qian Cao, Nemin Wu, Zeping
Liu, Lan Mu, Yang Song, Yiqun Xie, Ni Lao, et al .2025. LocDiffusion: Identify-
ing Locations on Earth by Diffusing in the Hilbert Space.Advances in Neural
Information Processing Systems (NeurIPS).
[34] Frederik Warburg, Soren Hauberg, Manuel Lopez-Antequera, Pau Gargallo, Yubin
Kuang, and Javier Civera. 2020. Mapillary street-level sequences: A dataset for
lifelong place recognition. InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR). 2626–2635.
[35] Tobias Weyand, Ilya Kostrikov, and James Philbin. 2016. Planet-photo geolocation
with convolutional neural networks. InProceedings of the European Conference
on Computer Vision (ECCV). Springer, 37–55.
[36] Biao Wu, Meng Fang, Ling Chen, Ke Xu, Tao Cheng, and Jun Wang. 2026. Vision-
Language Reasoning for Geolocalization: A Reinforcement Learning Approach.
Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)(2026).
[37] Yan-Tao Zheng, Ming Zhao, Yang Song, Hartwig Adam, Ulrich Buddemeier,
Alessandro Bissacco, Fernando Brucher, Tat-Seng Chua, and Hartmut Neven.
2009. Tour the world: building a web-scale landmark recognition engine. In
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
(CVPR). IEEE, 1085–1092.
[38] Zhongliang Zhou, Jielu Zhang, Zihan Guan, Mengxuan Hu, Ni Lao, Lan Mu,
Sheng Li, and Gengchen Mai. 2024. Img2Loc: Revisiting image geolocalization
using multi-modality foundation models and image-based retrieval-augmented
generation. InProceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval. 2749–2754.
[39] Sijie Zhu, Mubarak Shah, and Chen Chen. 2022. Transgeo: Transformer is all
you need for cross-view image geo-localization. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR). 1162–1171.