# SAR-RAG: ATR Visual Question Answering by Semantic Search, Retrieval, and MLLM Generation

**Authors**: David F. Ramirez, Tim Overman, Kristen Jaskie, Joe Marvin, Andreas Spanias

**Published**: 2026-02-04 16:23:16

**PDF URL**: [https://arxiv.org/pdf/2602.04712v1](https://arxiv.org/pdf/2602.04712v1)

## Abstract
We present a visual-context image retrieval-augmented generation (ImageRAG) assisted AI agent for automatic target recognition (ATR) of synthetic aperture radar (SAR). SAR is a remote sensing method used in defense and security applications to detect and monitor the positions of military vehicles, which may appear indistinguishable in images. Researchers have extensively studied SAR ATR to improve the differentiation and identification of vehicle types, characteristics, and measurements. Test examples can be compared with known vehicle target types to improve recognition tasks. New methods enhance the capabilities of neural networks, transformer attention, and multimodal large language models. An agentic AI method may be developed to utilize a defined set of tools, such as searching through a library of similar examples. Our proposed method, SAR Retrieval-Augmented Generation (SAR-RAG), combines a multimodal large language model (MLLM) with a vector database of semantic embeddings to support contextual search for image exemplars with known qualities. By recovering past image examples with known true target types, our SAR-RAG system can compare similar vehicle categories, achieving improved ATR prediction accuracy. We evaluate this through search and retrieval metrics, categorical classification accuracy, and numeric regression of vehicle dimensions. These metrics all show improvements when SAR-RAG is added to an MLLM baseline method as an attached ATR memory bank.

## Full Text


<!-- PDF content starts -->

SAR -RAG:  ATR Visual Question Answering by 
Semantic Search , Retrieval , and  MLLM Generation  
David F. Ramirez12, Tim Overman2, Kristen Jaskie2, Joe Marvin2, Andreas Spanias1 
1SenSIP Center, S chool of ECEE , Arizona State University, Tempe, Arizona  
2Prime Solutions Group Inc , Goodyear, Arizona  
1{dframire, spanias }@asu.edu  2{davidramirez, timoverman , kristenjaskie , joemarvin }@psg-inc.net
Abstract—We present  a visual -context image retrieval -
augmented generation (ImageRAG) assisted AI agent for 
automatic target recognition (ATR)  of synthetic aperture radar 
(SAR ). SAR is a remote sensing method  used in defense and 
security applications to detect and monitor the positions of 
military vehicles, which may appear indistinguishable in images. 
Researchers have extensively studied SAR ATR to improve the 
differentiation and identification of vehicle types, characteristics, 
and measurements. Test examples can be compared with known 
vehicle target types to improve recognition tasks. New methods  
enhance  the capabilities of neural networks, transformer 
attention, and multimodal large language models. An agentic AI 
method may be developed  to utilize  a defined set of tools, such as 
searching through  a library of similar examples. Our proposed 
method , SAR  Retrieval -Augmented  Generation (SAR -RAG), 
combines a multimodal large language model (MLLM) with a 
vector database of semantic embeddings to support contextual 
search for  image exemplars with known qualities. By recovering 
past image examples with known true target types, our SAR -
RAG  system can compare similar vehicle categories, achieving 
improved ATR prediction accuracy.  We evaluate this through 
search and retrieval metrics, categorical classification accuracy, 
and numeric regression of vehicle dimensions. These metrics all 
show improvements when SAR -RAG is added to a n MLLM 
baseline method as an attached ATR memory bank.  
Keywords —multimodal large language model, remote sensing , 
image retrieval -augmented generation, synthetic aperture radar, 
automatic target recognition  
I. INTRODUCTION  
Automatic target recognition (ATR) of vehicles in synthetic 
aperture radar (SAR) imagery is a challenging problem in 
remote sensing and defense . SAR -formatted images extrapolate 
spatial data from reflected radar energy, especially from 
metallic structures such as vehicles.  A scarcity of high -quality 
annotated SAR research  datasets has complicate d this task  [1-
2]. While traditional SAR ATR pipelines  built on machine 
learning [3], convolutional neural networks [ 4-5], and, more 
recently, transformer -based models [ 6-7] have made  progress 
under controlled experimental conditions, they often fail to 
generalize across diverse operational environments. These 
limitations, driven by data imbalance and scene variability, 
constrain both the accuracy and interpretability of current 
systems  [8-10]. Overcoming these challenges is essential in 
achieving robust, reliable, and explainable SAR ATR 
performance suitable for real -world military and intelligence 
applications.  Recent advances in multimodal large language models 
(MLLM) and multimodal learning have introduced retrieval -
augmented generation (RAG) as a promising framework for 
addressing these limitations  [11]. RAG enhances model 
reasoning by dynamically retrieving relevant knowledge —such 
as prior exemplars, contextual metadata, or semantic target 
descriptions —from an external database during inference. For 
SAR -based vehicle recognition, this matching to reference 
examples  has well -documented benefits  [12-13]: retrieved 
references can provide structural analogies, target -class priors, 
and spatial context that compensate for the scarcity and domain 
gaps in labeled SAR imagery. By fusing retrieved intelligence 
with SAR observations, our proposed RAG -driven systems can 
support more interpretable, data -efficient, and operationally 
robust ATR. This retrieval -augmented paradigm  establishes a 
foundation for next -generation SAR recognition models that 
align machine perception with human language.  
Recent research in remote sensing and geospatial analysis 
has begun to adopt RAG as an effective mechanism for 
integrating external knowledge into satellite imagery 
interpretation  [14-16]. Traditional deep learning models have 
focused primarily on image classification and object detection  
in large -scale scenes , but these approaches often lack 
contextual awareness and generalization across environments. 
In contrast, RAG introduces a dynamic retrieval process that 
supplements imagery analysis with relevant textual, spatial, or 
multimodal information during inference.  
Three significant trends have emerged from this shift. 
Firstly, knowledge -grounded analysis enhances models' ability 
to reference prior examples or domain metadata  [14], thereby 
improving interpretability and adaptability to new scenarios. 
Secondly, multimodal retrieval connects visual and linguistic 
information, facilitating richer semantic understanding for 
tasks such as captioning  [15], question answering, and change 
detection. Lastly, spatially aware retrieval incorporates 
geographic structures and relationships into reasoning, 
enabling models to consider location -dependent context  [16]. 
Collectively, these advancements indicate a move toward more 
intelligent, interpretable, and operationally robust geospatial AI 
systems.  
A. Problem Statement  
RAG introduces a robust framework for addressing the 
persistent limitations of SAR -based ATR. By integrating 
dynamic retrieval with generative reasoning, RAG enables 
models to augment learned SAR feature representations with 
externally sourced knowledge  regarding observed targets . In 

 
Fig. 1.  The SAR -RAG system diagram shows a continual learning loop.  
our approach, a retrieval module queries a curated repository 
containing SAR exemplars, vehicle signatures, and contextual 
intelligence. At the same time, a generative reasoning 
component generates  ATR predictions conditioned on both 
observed and retrieved evidence. This retrieval -guided 
mechanism grounds recognition in relevant prior knowledge, 
thereby mitigating hallucination, enhancing robustness in data -
scarce or novel conditions, and improving interpretability.  
While RAG introduces dynamic knowledge access into 
SAR -based ATR, its long -term effectiveness depends on the 
adaptability and continual relevance of its retrieval corpus and 
generative components. In real -world operational settings, new 
vehicle categories , sensor s, and environmental conditions 
frequently emerge, rendering static models inadequate for 
sustained performance. To overcome these limitations, we 
propose integrating continual learning into the RAG 
framework, enabling adaptive knowledge acquisition and 
incremental model refinement without catastrophic forgetting.  
Our method , Synthetic Aperture Radar Retrieval -
Augmented Generation (SAR -RAG), combines a multimodal 
large language model (MLLM) with a vector database of 
semantic embeddings to support contextual search for  vehicle  
exemplars with known qualities. By combining retrieval and 
generation, our SAR -RAG system can achieve greater data 
efficiency, contextual awareness, and operational 
maintainability , marking a significant step toward the 
development of explainable, mission -ready radar intelligence.  
In our continuous RAG system, both the retrieval database 
and the generative reasoning model evolve. The retrieval index 
continually incorporates newly acquired SAR samples, 
contextual metadata, and mission -specific intelligence. At the 
same time, the generative module incrementally updates its 
internal representations via parameter -efficient, memory -
constrained adaptation. This joint evolution allows the ATR 
system to preserve prior expertise while adapting to novel 
conditions. By coupling retrieval -augmented reasoning with 
continual learning, the proposed framework advances toward a 
form of sustainable adaptive intelligence  that continuously 
integrates new radar observations, maintains accumulated 
knowledge, and enhances generalization across diverse and 
evolving operational domains.  
II. BACKGROUND  
A. Addressing Machine Learning Challenges  
Large Language Models (LLM) have achieved exceptional 
performance in natural language understanding, reasoning, and 
generation. However, their knowledge remains inherently static and bounded by the information available at the time of 
machine learning training. This limitation poses a fundamental 
challenge for applications that require up -to-date, domain -
specific, or verifiable information. Although LLMs  generate 
coherent  text, their reliance on fixed neural network parameters 
often results in responses that are outdated, incomplete, or 
factually imprecise when applied to dynamic or specialized 
domains. Overcoming this constraint is essential for advancing 
toward more adaptive, trustworthy, and knowledge -aware 
language intelligence systems.  
RAG overcomes this limitation by introducing a retrieval 
stage before generation, enabling models to dynamically access 
external databases, document collections, or knowledge graphs 
at inference time  [11]. In this framework, retrieved information 
is fused with the model's internal representations, bridging the 
gap between parametric knowledge encoded in the model's 
parameters and non -parametric memory dynamically retrieved 
from up-to-date, external sources.  
This integration enables the model to ground its reasoning 
in verifiable and contextually relevant information rather than 
relying solely on learned associations. As a result, RAG -
enhanced systems produce outputs that are more factually 
accurate, contextually rich, and grounded in real -world data  
[11]. By coupling retrieval with generation, RAG transforms 
the LLM from a static text generator into an adaptive , 
evidence -based reasoning system  capable of continuous 
knowledge integration and trustworthy language generation . 
B. Multimodal Search, Retrieval, and Generation  
Traditional RAG frameworks have primarily concentrated 
on text -based retrieval, enabling LLM to enhance their 
responses with externally sourced, up -to-date written  
information. While this approach has proven effective for 
factual reasoning, it does not account for the extensive 
knowledge embedded in visual modalities, such as imagery, 
diagrams, and formatting . As visual data becomes increasingly 
central to modern artificial intelligence workflows, the 
limitations of purely textual retrieval have become more 
apparent. Text -based RAG systems lack the capacity to 
interpret or generate responses grounded in visual evidence, 
constraining their ability to reason about spatial relationships, 
object structures, and scene -level context. Addressing this gap 
requires extending RAG to integrate visual information directly 
into the retrieval and generation process, thereby enabling 
multimodal reasoning that unifies language and perception.  
ImageRAG [17] extends the traditional RAG framework by 
incorporating visual information as a central component of 
retrieval and reasoning [ 18-20]. These RAG systems index and 
retrieve visual content , such as images, maps, or remote -
sensing scenes  [14-16], alongside textual evidence to provide 
richer, multimodal context. Visual data is encoded into high -
dimensional numeric embeddings using vision encoders or 
multimodal models and stored in vector databases that enable 
efficient similarity -based retrieval.  
During inference, the system retrieves visual instances that 
are semantically similar  to the input query and integrates them 
into the generative reasoning process. This fusion enables the 
model to reason across visual and textual modalities jointly , 

interpret spatial relationships, and generate explanations 
grounded in observable evidence. By aligning visual retrieval 
with language generation, ImageRAG enhances factual 
grounding, contextual richness, and cross -modal understanding  
[17], thereby advancing toward truly evidence -based, visually 
grounded intelligence applicable to domains such as geospatial 
analysis, defense, and security . 
C. Remote Sensing RAG  
In the domain  of remote sensing imagery and geospatial 
analysis, the adoption of RAG techniques is developing  as a 
transformative trend. A key work is “RS-RAG: Bridging 
Remote Sensing Imagery and Comprehensive Knowledge with 
a Multimodal Dataset and Retrieval‑Augmented Generation 
Model ” [14]. This research  introduces a multimodal knowledge 
database of high-resolution satellite imagery and extensive 
textual d escriptions of over 14,000 landmarks globally . This 
data source is accessed by a retrieval -augmented vision -
language model (VLM) for captioning, classification, and 
visual question answering (VQA ) tasks. The authors report 
substantial gains over baseline VLM that lack retrieval 
components.  Another promising published work, “Enhancing 
Ultra High Resolution Remote Sensing Imagery Analysis with 
ImageRAG ” [15], addresses the challenge of ultra -high-
resolution (UHR) imagery by using a retrieval mechanism to 
select relevant image patches for analysis. This ImageRAG 
framework enables a model to focus on critical context within 
large scenes, thereby mitigating memory/compute issues while 
improving interpretive accuracy.  Also concentrated  on 
geospatial reasoning,  the “Spatial‑RAG: Spatial Retrieval 
Augmented Generation for Real‑World Spatial Reasoning 
Questions” framework [16] extends RAG with a geospatial 
database, combining spatial location retrieval with semantic 
information to answer location -based queries using a n LLM. 
RAG is evolving beyond purely image - or text -based retrieval 
to more structured spatial retrieval for geospatial tasks.  
III. METHODS  
The ImageRAG paradigm offers a novel approach to 
enhancing understanding and reasoning with SAR imagery. By 
integrating visual retrieval with generative reasoning, our 
SAR -RAG  system identifies and retrieves pertinent SAR 
image exemplars, object templates, or scene segments from a 
curated database, then merges them with the current query 
context to enrich the model's reasoning. In the context  of VQA, 
this framework enables an AI system to accurately respond to 
textual inquiries about SAR scenes while grounding its 
answers in retrieved visual evidence from analogous imagery . 
Fig. 1 shows a system diagram, with data flow arrows and 
processing steps.  
A. Data Corpus & Vector Database  
The Moving and Stationary Target Acquisition and 
Recognition (MSTAR) dataset serves as a widely recognized 
benchmark for evaluating SAR –based ATR  [1-2]. This 
comprises X -band SAR imagery of several  military ground 
vehicles collected under controlled conditions with varying 
depression angles and aspect orientations. Each sample 
includes a greyscale SAR  image chip, a complex -phase raster, and detailed metadata, including  target class, serial number, 
collection  angle, azimuth orientation, and configuration 
parameters. These characteristics make MSTAR particularly 
suitable for constructing a retrieval -augmented corpus that 
supports multimodal reasoning.  We combine portions of the 
MSTAR Mixed Targets and T -72 Variants datasets . 
Specifically, we utilize the 2S1, BRDM -2, BTR -60, D7, 
SLICY, T -62, T -72, ZIL -131, and ZSU -23-4 vehicle types. 
This results in a total of 14,108  SAR image chips with 
metadata. Note that the T -72 vehicle type is overrepresented in 
the data, resulting in a 10:1 imbalance versus  the least -
represented target. We perform a 50/50 stratified random split  
of this data to form training and validation sets , repeating this 
process several times to form multiple experiments.  
To develop the ImageRAG corpus, we process  the MSTAR 
training subset target images to conform to  a multimodal vector 
database .   Each image chip, representing a specific vehicle and 
sensor configuration, is encoded with a standard  embedding 
model  utilizing the LlamaIndex framework [21] . The resulting 
embeddings are indexed in a Qdrant vector database [22] to 
enable similarity -based retrieval.  In parallel, associated vehicle 
metadata is also stored  in the Qdrant database , capturing 
written attributes such as target type, vehicle characteristics, 
measurements, depression angle, azimuth angle, and condition, 
with each metadata entry linking to its corresponding image via 
a unique identifier.  
This hybrid architecture, combining visual embeddings and 
structured metadata, enables both content -based retrieval and 
context -aware filtering. During inference, when presented with 
a query image or a natural -language question, the system 
retrieves semantically similar SAR exemplars along with their 
contextual metadata to ground the model's reasoning. This 
integrated design supports downstream tasks such as VQA, 
target recognition, and explanation generation, thereby 
enhancing interpretability and ensuring that ground -truth SAR 
observations explicitly support system outputs . 
B. Domain -Specific Cross -Modal Alignment  
We use a variant of the Qwen2 vision -language transformer 
neural network [23] to generate image and language 
embeddings. This visual encoder transforms SAR imagery into 
semantic embeddings that preserve key structural and radar 
scattering characteristics relevant to retrieval and reasoning. 
Unlike optical imagery, SAR data exhibit unique properties 
such as speckle noise, anisotropic backscattering, and sensor -
dependent distortions, which pose significant challenges for 
feature extraction. Consequently, vision encoders pretrained on 
natural images often perform suboptimally  when applied 
directly to SAR data, necessitating domain -specific adaptation.   
To address this, we fine -tune the image encoder on SAR 
data using supervised fine -tuning . This enables the model to 
learn class -discriminative radar features . Through this 
optimization , the encoder learns a radar -specific representation 
space that balances  visual discrimination and semantic 
similarity.  We follow model -training steps similar to our prior 
work described in “Towards a Large Language -Vision 
Question Answering Model  for MSTAR Automatic Target 
Recognition ” [6].  

TABLE I.  EVALUATION RESULTS  
SAR -RAG Retrieval & Visual Question Answering Benchmark  
Evaluation Metric  SAR -RAG  
(mean)  Baseline 
(mean)  Random 
(weighted)  
Retrieval  
Accuracy @ 1 -shot 77.72% 
NA 21.11%  
Precision @  5-shot 74.39% 
Any Correct @ 5-shot 93.54% 69.44%  
All Correct @ 3 -shot 61.58% 0.94% 
All Correct @ 5 -shot 53.54% 0.042% 
Vehicle Classification  (accuracy)  
Type  99.24% 99.04%  
20.95% 
Descriptive Qualities Set  98.79% 98.47% 
Mounted Weapon Detection  100.0%  100.0%  52.19% 
Vehicle Weight (metric tons)  
MAE  0.428  0.530 10.41  
RMSE  0.891  1.124  15.01  
MAPE  1.24%  1.57%  75.45%  
Vehicle Dimensions  (meters)  
MAE  0.2639  0.3328 0.9584  
RMSE  0.6665  0.8407  0.9068  
MAPE  10.39%  13.11%  20.41%  
 
We train an MLLM with an attached SAR -RAG module 
for several VQA tasks. Similar to our prior work [6], we utilize 
the LLaVA -Next v1.6 Mistral -7B MLLM and parameter -
efficient fine -tuning.  
C. Evaluation Metrics  
We evaluate the performance of our proposed SAR -RAG 
system using a combination of retrieval, classification, VQA, 
and regression  metrics to assess accuracy and robustness 
comprehensively . The SAR -RAG framework is compared with 
a baseline MLLM method trained similarly but without an 
ImageRAG component. These two methods are compared 
against the theoretical performance of arbitrary blind guesses 
weighted by the known distribution of the vehicle types. This 
comparison sets a lower bound on performance for the 
complex evaluations that follow:  
1) Retrieval Performance : Retrieval quality measures the 
system's ability to identify relevant SAR exemplars from the 
database given a query image.  Given a validation image, we 
recover the five most similar image vectors using cosign 
similarity. The types of these five comparison vehicles are 
compared with the evaluation sample. The closest matching 
vector is evaluated on Accuracy.  The Precision metric then 
measures the percentage of accurate retrieval among all five. 
We also check whether any of the five are correct and whether 
the top several samples are all correct. The MLLM baseline 
does not include a retrieval module and cannot be evaluated.  2) Automatic Target Recognition : For classification -
oriented evaluation s, accuracy  measures the  correctness of 
target identification.  We also evaluate the models’ ability to 
generate a set of descriptive qualities associated with a 
vehicle, similar to our prior work [6].  We assess  the model’s 
ability to detect a mounted weapon system as a binary 
prediction problem . 
3) Quantitative VQA Metrics : Natural language outputs 
generated by an MLLM can be challenging  to evaluate. 
English grammar, word choice, semantic meaning, and factual 
correctness must be decoupled for the most precise evaluation. 
We focus on assessing  factual correctness, specifically the 
accuracy of numeric values. Continuous numeric attributes, 
such as the vehicle ’s weight, length, width, and height, are 
evaluated using Mean Absolute Error (MAE), Root Mean 
Squared Error (RMSE), and Mean Absolute Percentage  Error 
(MAPE).  The size predictions are averaged together and are 
measured in meters. We explore this quantitative spatial 
reasoning in greater detail in our future work.  
IV. RESULTS  
In Table 1, SAR -RAG shows good performance on both 
database lookup matching and improvements across all VQA 
benchmarks.  
The ImageRAG retrieval module often returned a matching 
vehicle type, which is ideal. Despite the visual greyscale 
similarity among SAR images , the embedding method 
produced  vectors that clustered  similar vehicle types.  Note  how 
often all five vectors match a given query. This is evident in 
the t-distributed Stochastic Neighbor Embedding  (t-SNE) 
vector dimensionality reduction algorithm and visualization in 
Fig. 2. Also of note is the separation of the SLICY target from 
the true vehicles.  It is well documented  that his radar reflection 
simulation target appears very different from the others, as 
shown in the vector embedding.  
The overall accuracy of the retrieval rates positively 
influences the generation of correct answers in subsequent 
evaluations, as the ground -truth data provides context for the 
MLLM's decision -making.  This is most evident in the 
quantitative reasoning metrics. The SAR -RAG enhanced 
method is trained to consider similar vehicles and their known 
accurate  measurements. This reduces the chances of a serious 
hallucination, which we have documented as a n issue in the 
baseline model . When the predicted value is erroneously off by 
an order of magnitude, this greatly inflates the error metrics. 
By providing a readily accessible source of information for the 
MLLM, the likelihood of substantial error decreases 
significantly .  
Finally, we observe minor improvements in the ATR 
classification metrics of the SAR -RAG enhanced method 
relative to  the baseline. Vehicle type prediction is slightly 
improved on average, although it is already very accurate . The 
set of d escriptive qualities can be predicted precisely if the 
actual  vehicle type is known. Without first inferring a vehicle 
type, generating  all the vehicle's qualities is more prone to 
error . Similarly, the SAR -RAG retrieval is not perfect and can 
introduce errors if the incorrect vectors match.  

 
Fig. 2.  The t-SNE  dimensionality reduction algorithm shows clusters of 
similar SAR image embeddings of different vehicle types.  
V. CONCLUSIONS  
SAR -RAG is a new technique for referencing a database of 
ATR targets before an MLLM predicts the answer for  a VQA 
task. This additional information improves prediction 
performance across  all tested benchmark tasks. We evaluated 
the retrieval efficacy of this vector search  by retrieving the five 
nearest embeddings to a query image using  cosine  similarity. 
Our trained MLLM improved slightly with our proposed 
ImageRAG  system , reaching nearly perfect accuracy. Finally , 
we benchmarked against several quantitative VQA tasks that 
infer  the size of the target vehicles. This was greatly improved 
by the inclusion of the SAR -RAG system , which reduced rare 
but persistent significant  errors  in the baseline model.  
In addition to  improved accuracy, a RAG system offers 
other benefits.  RAG reduces the amount of retraining required 
for transformer methods, since updates can be applied to the 
knowledge database rather than  costly AI retraining. 
Additionally, filtering mechanisms can be added to the vector 
lookup to focus on specific environmental conditions, for 
example , only retrieving similar SAR images from a particular 
sensor elevation  angle. RAG can also be extended to use more 
advanced database varieties. Graph databases or knowledge 
graphs [24] can be structured for a RAG process, unlocking 
even greater context discovery by the MLLM AI agent.  
SAR -RAG is another compelling extension of MLLM 
methods and VQA tasks. Our work to date has extensively 
used the MSTAR and SAMPLE datasets . We have extended 
these traditional SAR ATR datasets with modern VQA text 
sequences to train MLLM for novel tasks. Our future work 
explores  additional quantitative ATR VQA tasks and new data 
sources [25]. Researchers should consider what  other 
application domains can benefit from powerful new AI models 
tailored to specific use cases.  
ACKNOWLEDGMENT  
This work is supported by Prime Solutions Group Inc. and 
the ASU Sensor, Signal, and Information Processing (SenSIP) 
Center . Several A.I. tools were used during our literature 
review , proofreading , and editing , including AI2 Astra, 
ChatGPT, Google Gemini, and Grammarly.  REFERENCES  
[1] E. R. Keydel, Shung Wu Lee, and John T. Moore, "MSTAR Extended 
Operating Conditions: A Tutorial," Proc. SPIE 2757, Algorithms for 
Synthetic Aperture Radar Imagery III, 10 June 1996.  
[2] DARPA and AFRL, Sep. 1995, "Moving and Stationary Target 
Acquisition and Recognition (MSTAR) Public Release," Sensor Data 
Management System. [Online]. https://www.sdms.afrl.af.mil/ index.php? 
collection=mstar.  
[3] M. D. Scherreik and B. D. Rigling, "Open Set Recognition for 
Automatic Target Classification with Rejection," in IEEE Transactions 
on Aerospace and Electronic Systems, vol. 52, no. 2, pp. 632 -642, 2016.  
[4] S. R. Sellers, P. J. Collins , and J. A. Jackson, "Augmenting Simulations 
for SAR ATR Neural Network Training," IEEE International Radar  
Conference (RADAR), 2020, pp. 309 -314. 
[5] S. Chen, H. Wang, F. Xu, and Y. -Q. Jin, "Target Classification Using 
the Deep Convolutional Networks for SAR Images," in IEEE 
Transactions on Geoscience and Remote Sensing, vol. 54, no. 8, pp. 
4806 -4817, Aug. 2016.  
[6] D. F. Ramirez, T. L. Overman, K. Jaskie, M. Kleine, and A. Spanias, 
“Towards a Large Language -Vision Question Answering Model for 
MSTAR Automatic Target Recognition,” in Automatic Target 
Recognition XXXV, vol. 13463, International Society for Optics and 
Photonics, SPIE, 2025.  
[7] G. Zhao et al., "Towards SAR Automatic Target Recognition: Multi -
Category SAR Image Classification Based on Light Weight Vision 
Transformer," 2024 21st Annual International Conference on Privacy, 
Security and Trust (PST), Sydney, Australia, 2024, pp. 1 -6. 
[8] M. A. Saville, J. A. Jackson, and D. F. Fuller, "Rethinking Vehicle 
Classification with Wide -Angle Polarimetric SAR," in IEEE Aerospace 
and Electronic Systems Magazine, vol. 29, no. 1, pp. 41 -49, Jan. 2014.  
[9] R. W. Jansen, R. G. Raj, L. Rosenberg , and M. A. Sletten, "Practical 
Multichannel SAR Imaging in the Maritime Environment," in IEEE 
Transactions on Geoscience and Remote Sensing, vol. 56, no. 7, pp. 
4025 -4036, July 2018 . 
[10] M. S. Greco and F. Gini, "Statistical Analysis of High -Resolution SAR 
Ground Clutter Data," in IEEE Transactions on Geoscience and Remote 
Sensing, vol. 45, no. 3, pp. 566 -575, March 2007.  
[11] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. 
Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela , 
“Retrieval -Augmented Generation for Knowledge -Intensive NLP 
Tasks,” Advances in Neural Information Processing Systems  (NeurIPS) , 
vol. 33, pp. 9459 -9474 , 2020 . 
[12] E. E. Laubie, B. D. Rigling, and R. P. Penno, "Decreased Probability of 
Error in Template -Matching Classification Using Aspect -Diverse 
Bistatic SAR," in IEEE Transactions on Aerospace and Electronic 
Systems, vol. 54, no. 4, pp. 1862 -1870, Aug. 2018.  
[13] J. A. Jackson, "Sparse Regularization Effects on Radar ATR," IEEE 
RADAR, 2025, Atlanta, GA, USA, pp. 1 -5. 
[14] C. Wen, Y. Lin, X. Qu, N. Li, Y. Liao, H. Lin, and X.  Li, “RS-RAG: 
Bridging Remote Sensing Imagery and Comprehensive Knowledge with 
a Multi -Modal Dataset and Retrieval -Augmented Generation Model ,” 
arXiv: 2504.04988 [cs .CV], Apr. 2025 . 
[15] Z. Zhang, H. Shen, T. Zhao, Z. Guan, B. Chen, Y. Wang, X. Jia, Y. Cai, 
Y. Shang, and J. Yin , "Enhancing Ultrahigh Resolution Remote Sensing 
Imagery Analysis With ImageRAG: A New Framework," in IEEE 
Geoscience and Remote Sensing Magazine, vol. 13, no. 3, pp. 369 -394, 
Sept. 2025 . 
[16] D. Yu, R. Bao, G. Mai, and L.  Zhao , “Spatial -RAG : Spatial Retrieval 
Augmented Generation for Real-World Spatial Reasoning Questions ,” 
arXiv: 2502.18470 [cs.IR], Feb. 2025 . 
[17] R. Shalev -Arkushin , R. Gal, A. H. Bermano , and O. Fried , “ImageRAG: 
Dynamic Image Retrieval for Reference -Guided Image Generation ,” 
arXiv:  2502.09411  [cs.CV ], Feb 2025 . 
[18] S. Sarto, M. Cornia, L. Baraldi, and R.  Cucchiara , “Retrieval -
Augmented Transformer for Image Captioning ,” IEEE Proceedings of 
the 19th International Conference on Content -based Multimedia 
Indexing  (CBMI) , pp. 1–7, Oct. 2022 . 

[19] W. Chen, H. Hu, X. Chen, P. Verga, and W.  W. Cohen , “MuRAG: 
Multimodal Retrieval -Augmented Generator for Open Question 
Answering over Images and Text ,” arXiv:  2210.02928 [cs.CL] , 2022.  
[20] S. Bag, A. Gupta, R. Kaushik , and C. Jain, "RAG Beyond Text: 
Enhancing Image Retrieval in RAG Systems," 2024 International 
Conference on Electrical, Computer and Energy Technologies 
(ICECET ), 2024, pp. 1 -6. 
[21] J. Liu, “ LlamaIndex ,” 2022. [Source Code]. Available: github.com/  
jerryjliu/llama_index . 
[22] Qdrant. Available: https://qdrant.tech/. Accessed: Nov.  18, 2025 . [23] Qwen Team  and Alibaba Group , “Qwen2.5 -VL Technical Report ,” 
arXiv:  2502.13923  [cs.CV] , Feb. 2025.  
[24] J. Liang, S. Hou, H. Jiao, Y. Qing, A. Zhao, Z. Shen, L. Xiang, and H. 
Wu, “GeoGraphRAG: A Graph-Based Retrieval -Augmented Generation 
Approach for Empowering Large Language Models in Automated 
Geospatial Modeling, ” International Journal of Applied Earth 
Observation and Geoinformation , vol. 142, no. 104712 , Aug.  2025 . 
[25] W. Li, W. Yang, Y. Hou, L. Liu, Y. Liu, and X. Li, "SARATR -X: 
Toward Building a Foundation Model for SAR Target Recognition," in 
IEEE Transactions on Image Processing, vol. 34, pp. 869 -884, 2025.  
 