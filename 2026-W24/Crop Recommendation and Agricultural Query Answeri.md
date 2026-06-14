# Crop Recommendation and Agricultural Query Answering System Using Spatio-Temporal Graph Neural Networks and Hybrid Retrieval Augmentation

**Authors**: Prajwal Thapa, Yagya Raj Pandeya

**Published**: 2026-06-08 07:51:34

**PDF URL**: [https://arxiv.org/pdf/2606.09160v1](https://arxiv.org/pdf/2606.09160v1)

## Abstract
This paper presents a unified system designed to support precision agriculture by integrating advanced weather prediction, crop recommendation, and a question-answering tool for farmers. We propose two deep learning models -- a Transformer-based Graph Neural Network and a Spatio-Temporal Graph Convolutional Network (STGCN) -- to forecast weather conditions for the next 30 days using data from 1,359 locations in Nepal. The STGCN outperforms the Transformer-based model in accuracy (MSE ~0.011 vs. 0.013), effectively modeling both spatial and temporal dependencies in climate data. These predictions are combined with static soil properties such as pH, moisture, and organic content to generate localized crop recommendations through a scoring algorithm that matches each crop's optimal growing conditions. Additionally, we develop a Retrieval-Augmented Generation (RAG) chatbot that leverages domain-specific agricultural documents to answer farmers' questions in natural language. The entire system is deployed via a mobile application, offering real-time suggestions and conversational support. User feedback confirms the system's usability and relevance, especially in rural settings where personalized farming guidance is limited. Overall, our approach demonstrates how combining machine learning models with local agricultural data can empower farmers with actionable insights, promoting more informed decisions, better crop yields, and increased resilience to climate variability.

## Full Text


<!-- PDF content starts -->

Crop Recommendation and Agricultural Query
Answering System Using Spatio-T emporal Graph
Neural Networks and Hybrid Retrieval
Augmentation
Prajwal Thapa
Kathmandu University
Dhulikhel, Nepal
prazzwalthapa87@gmail.comY agya Raj Pandyea
Kathmandu University
Dhulikhel, Nepal
yagya.pandeya@ku.edu.np
Abstract—This paper presents a unified system designed to
support precision agriculture by integrating advanced weather
prediction, crop recommendation, and a question-answering
tool for farmers. W e propose two deep learning models—
a T ransformer-based Graph Neural Network and a Spatio-
T emporal Graph Convolutional Network (STGCN)—to fore-
cast weather conditions for the next 30 days using data
from 1,359 locations in Nepal. The STGCN outperforms the
T ransformer-based model in accuracy (MSE ≈ 0.011 vs. 0.013),
effectively modeling both spatial and temporal dependencies
in climate data. These predictions are combined with static
soil properties such as pH, moisture, and organic content to
generate localized crop recommendations through a scoring
algorithm that matches each crop’s optimal growing conditions.
Additionally , we develop a Retrieval-Augmented Generation
(RAG) chatbot that leverages domain-specific agricultural doc-
uments to answer farmers’ questions in natural language. The
entire system is deployed via a mobile application, offering real-
time suggestions and conversational support. User feedback
confirms the system’s usability and relevance, especially in
rural settings where personalized farming guidance is limited.
Overall, our approach demonstrates how combining machine
learning models with local agricultural data can empower
farmers with actionable insights, promoting more informed
decisions, better crop yields, and increased resilience to climate
variability .
Index T erms—Agriculture, W eather Prediction, Soil, Graph
Neural Networks, Mobile Application, Retrieval Augmented
Generation (RAG), Chat
I. Introduction
In Nepal, approximately 66% of the population is
involved in agriculture, which accounts for about a third
of the country’s GDP and plays a crucial role in the
national economy . This sector provides a wide range of
job opportunities, from farming to small-scale businesses.
However, Nepal faces challenges such as food insecurity
and frequent natural disasters, including floods, droughts,
landslides, earthquakes, diseases, and pest outbreaks. The
agriculture sector is predominantly characterized by sub-
sistence farming, leading to low productivity and output
of agricultural goods.T raditionally , Nepalese farmers have relied on experi-
ence, historical weather patterns, and basic meteorological
forecasts to make crop-related decisions. However, these
methods often lack precision and fail to account for
rapid climate changes and extreme weather events that
have become more frequent in recent years due to global
warming. The need for more sophisticated, data-driven
approaches to agricultural decision-making has led to the
development of advanced technologies, such as machine
learning (ML) and artificial intelligence (AI).
Machine learning techniques have been widely used in
agriculture for tasks like yield prediction [ 1], pest detec-
tion, and disease diagnosis. Recently , the advent of deep
learning, particularly graph-based neural network and
transformer-based models, has opened new possibilities
for weather prediction and crop recommendation systems.
Graph Neural Networks (GNN) [ 2] and T ransformers
[3], initially developed for natural language processing
tasks and molecular chemistry , molecular biology , pat-
tern recognition, and data mining, have demonstrated
exceptional capabilities in handling sequential data, mak-
ing them suitable for time-series forecasting, including
weather prediction.
The core idea behind using spatio-temporal graph and
transformer-based graph models for weather prediction
is their ability to capture long-range dependencies and
complex patterns in large datasets. These models can
analyze historical weather data, learn from it, and make
accurate predictions about future weather conditions. By
integrating these predictions with other relevant agricul-
tural data, such as soil properties and crop characteristics,
a comprehensive crop recommendation system can be
developed.
Such a system can provide farmers with timely and
precise recommendations on which crops to plant based
on predicted weather conditions, optimizing yields and
reducing losses. Moreover, it can help mitigate the impacts
of adverse weather events, enhancing the resilience ofarXiv:2606.09160v1  [cs.LG]  8 Jun 2026

agricultural systems to climate variability and change.
In addition to the recommendation system, we also
developed a knowledge-based RAG system designed to
handle queries from farmers regarding various agricul-
tural information. This includes details about crops, pest
control, ongoing subsidies, and other relevant topics. T o
enhance the system’s performance, we employed a hybrid
approach using SPLADE and dense-sparse vectors for
querying. SPLADE is particularly effective in retrieving
sparse representations, which can capture relevant infor-
mation with minimal computational overhead, and sparse
vectors allow for more eﬀicient matching of queries to
the knowledge base. Dense vectors help to capture the
semantic meanings. This approach ensures that farmers
receive accurate and timely information, empowering them
to take more informed actions in their farming practices,
ultimately contributing to better outcomes in agriculture.
This thesis explores the development of a crop rec-
ommendation system that leverages transformer-based
models for weather prediction. By combining advanced
machine learning techniques with agricultural data, this
research aims to create a tool that supports farmers
in making informed decisions, ultimately contributing to
more sustainable and productive agricultural practices.
II. Related W ork
This chapter provides a detailed overview of the sys-
tem’s functionality , outlining its core components and
the methodologies adopted for weather prediction and
crop recommendation. It includes a review of the current
research landscape and existing studies in transformer-
based weather forecasting and agricultural decision sup-
port systems. F urthermore, the chapter identifies and
discusses the limitations and challenges present in current
systems, highlighting areas ripe for improvement to better
assist farmers in making informed decisions grounded in
precise weather forecasts and integrated soil data.
The increasing emphasis on precision agriculture has
fueled significant research efforts aimed at developing crop
recommendation systems that effectively integrate diverse
environmental and geographical factors. These systems
have evolved considerably , transitioning from simpler rule-
based approaches to more sophisticated machine learn-
ing and deep learning models capable of incorporating
dynamic variables like weather conditions, soil proper-
ties, and spatial information. F or a country like Nepal,
characterized by its varied climatic zones and soil types
across 1359 distinct locations, such advanced systems are
indispensable for maximizing agricultural output.
Advances in Crop Recommendation Systems
The progression in developing crop recommendation
systems is marked by the increasing incorporation of
multiple data sources, including weather patterns, soil
attributes, and spatial dependencies, into predictive mod-
els. Early research often relied on single-variable models,whereas recent efforts focus on integrating diverse datasets
to achieve more accurate and holistic predictions.
W eather Prediction in Agriculture: W eather prediction
holds a central position in crop recommendation, given
that environmental parameters such as temperature, rain-
fall, and humidity directly influence crop growth and
development. [ 4] presented one of the initial attempts
to integrate weather prediction into agricultural decision-
making using a linear regression model, though this model
faced diﬀiculties in handling non-linear weather patterns.
Subsequent, more advanced models, including those de-
scribed by [ 5], utilized algorithms like Random F orest and
Support V ector Machines (SVM) to account for climate
variability , demonstrating improved accuracy in crop yield
prediction. More recently , deep learning methodologies,
particularly transformer models, have been adapted for
weather forecasting. [ 6] introduced a transformer-based
model that achieved significantly enhanced prediction
of long-term weather patterns, attributed to its strong
capability to capture temporal dependencies over extended
sequences.
Soil Data and Crop Recommendation: The critical im-
portance of soil data, encompassing factors like pH levels,
moisture content, and nutrient availability , has been con-
sistently highlighted across various crop recommendation
studies. [ 7] implemented a system that relied exclusively
on soil attributes to recommend suitable crops, employing
decision trees for soil type classification. However, this
method was inherently limited by its inability to account
for dynamic environmental factors such as fluctuating
weather conditions. [ 8] advanced the field by integrating
both weather and soil data into a hybrid machine learning
model, which resulted in improved accuracy , particularly
beneficial for specific regions with limited data availability .
Despite this progress, many existing models still do not
dynamically integrate temporal changes in soil properties,
which is a crucial aspect for providing accurate recom-
mendations over time.
Spatio-T emporal Data Integration: Spatio-temporal
data plays a vital role in the accurate modeling of
agricultural systems, especially within geographically di-
verse countries like Nepal. Spatial data pertains to the
geographical characteristics of a given location, while
temporal data tracks changes in conditions over time.
[9] developed Spatio-T emporal Graph Neural Networks
(STGNNs) specifically to analyze spatial and temporal
data concurrently , proving useful for predicting crop
yields across different geographical regions. This approach
effectively addressed the challenge of modeling dynamic
environmental conditions by incorporating both temporal
sequences and spatial relationships into the analysis.
Similarly , [ 10] extended this work, applying STGCNs
to predict environmental conditions directly relevant to
agriculture, further emphasizing the significance of spatio-
temporal modeling in this domain. Nevertheless, the
specific application of these advanced techniques directly

for crop recommendation in agriculture remains an area
with limited exploration.
Graph Neural Networks in Agriculture: Graph Neural
Networks (GNNs) have emerged as powerful tools particu-
larly well-suited for handling spatial dependencies inherent
in agricultural data, especially in environments involving
multiple interconnected locations. [ 11] successfully applied
GNNs to predict crop yields across a network of locations,
effectively capturing the spatial relationships and depen-
dencies between different regions. While beneficial for
modeling regions with similar soil and weather character-
istics, this approach did not incorporate temporal changes,
which are fundamental for accurate and dynamic recom-
mendations. [ 12] addressed this limitation by integrating
both spatial and temporal data into a GNN model, leading
to significant improvements in the performance of crop
recommendation systems across varying environmental
zones.
T ransformers in Agricultural Systems: Originally de-
veloped for state-of-the-art natural language processing
tasks, transformer models have been successfully adapted
for time-series forecasting applications within the agricul-
tural sector. The attention mechanism, a key innovation
introduced by [ 3], has made transformers exceptionally
useful for predicting long-term weather patterns due to
their capability to focus on the most relevant parts of a
sequence while disregarding less important information.
[13] developed a transformer-based model specifically for
predicting crop yields, demonstrating that the model could
effectively process and learn from large-scale agricultural
datasets. This work underscored the potential for inte-
grating transformers into agricultural systems, although
further research is needed to combine the strengths of
transformers with GNNs to build more robust and com-
prehensive crop recommendation systems.
Summary of Related W ork
In T able I, a summary of the major contributions
from related studies in crop recommendation systems is
presented, highlighting their various data sources, method-
ologies, and identified limitations.
Research Gap
Despite the considerable advancements made in devel-
oping crop recommendation systems, several significant
gaps remain unaddressed in the existing body of research.
Firstly , a common limitation is that most systems tend
to focus primarily on either weather prediction or the
analysis of soil data in isolation, failing to integrate both
crucial data sources within a unified modeling framework
that effectively accounts for spatio-temporal dependencies.
While a few studies, notably [ 9] and [ 12], have successfully
combined spatio-temporal data analysis with GNNs, these
specific applications have not yet been adapted or tailored
for direct use in agricultural crop recommendation sys-
tems. F urthermore, while transformer models have clearlydemonstrated their effectiveness in time-series forecasting,
particularly for weather prediction tasks, their integration
with GNNs to simultaneously handle both spatial rela-
tionships and temporal dependencies within the context
of agricultural crop recommendation systems remains an
area that is significantly underexplored. Nepal’s unique
and highly diverse geographical landscape, encompassing
data from 1359 distinct locations, presents a specific
challenge for traditional models, as they often struggle
to effectively account for both localized microclimates
and broader regional or global trends in weather and
soil conditions concurrently . This research is specifically
designed to bridge these identified gaps by proposing
and implementing a novel system that integrates the
capabilities of transformer models for accurate weather
prediction with Spatio-T emporal Graph Convolutional
Networks (ST-GCNs) for comprehensive soil and spatial
data analysis, a system specifically tailored to meet the
distinct needs and challenges of Nepalese agriculture.
III. Methodology
This research employs advanced machine learning tech-
niques for weather prediction and crop recommendation.
The methodology consists of two main parts: weather
prediction using Graph Neural Networks (GNNs) and
T ransformer-based Graph models, and crop recommen-
dation based on predicted weather and soil data.
A. Datasets
F our datasets were utilized: location, weather, soil, and
crop data.
1) Location Data: Location data for 1359 agricultural
regions in Nepal was collected manually using [ 14]. This
dataset [ 15] includes Longitude, Latitude, Altitude, and
Location Name for diverse farming zones (see 1).
2) W eather Data: A comprehensive weather dataset
[16] covering 1359 locations over 42 years was ob-
tained from NASA’s [ 17]. Key variables include T emper-
ature (T2M), Precipitation (PRECTOTCORR), Humid-
ity (RH2M), W et Bulb T emperature (T2MWET), Wind
speed (WS10M), and Earth Skin T emperature (TS).
Preprocessing involved correlation testing to select the
top 6 features most relevant to T emperature, Humidity ,
and Precipitation. Data was then standardized using the
formula:
zi=xi−µi
σi
where xiis the original value, µiis the mean, and σiis
the standard deviation of the feature. This ensures features
contribute equally and improves model performance.
T emporal snapshots were created with a 120-day lag to
capture seasonal variability . Locations were represented
as nodes in a graph structure. Edges were defined based
on either geodesic distance less than 15 km (using the
Haversine formula) or altitude difference less than 50
meters. Edge weights were inversely proportional to the

Author(s) Data Sources Methods Used Limitations
[4] W eather data (rainfall,
temperature)Linear Regression Poor performance on non-
linear data, no soil or spatial
analysis.
[5] Climate data (rainfall,
humidity)Random F orest, SVM Improved accuracy but no inte-
gration of soil or spatial data.
[7] Soil data (pH, nutri-
ents)Decision T ree Limited to static soil data, no
temporal or weather integra-
tion.
[8] W eather and soil data Hybrid Model (Soil +
W eather)No spatio-temporal considera-
tion, regional application only .
[6] W eather time-series
dataT ransformer for time-
series forecastingF ocused solely on weather
data, without spatial or soil
integration.
[9] Spatio-temporal
weather and yield
dataSpatio-T emporal
Graph Neural
Networks (STGNN)Limited application in agricul-
ture, focused on environmental
data.
[11 ] Spatial yield and soil
dataGraph Neural
Networks (GNNs)Lacked temporal analysis, fo-
cusing only on spatial depen-
dencies.
[12 ] W eather, soil, and tem-
poral dataGNN integrated with
environmental dataDid not include transformer-
based prediction for long-term
trends.
[13 ] W eather data for crop
yieldT ransformer-based
predictionOnly integrated weather data,
no spatial or soil inclusion.
[10 ] Spatio-temporal envi-
ronmental dataSpatio-T emporal
Graph Convolutional
NetworksF ocused on environmental fac-
tors, limited crop recommenda-
tion use.
T ABLE I
Summary of Contributions in Crop Recommendation Systems
   
  
Fig. 1. Locations selected all over Nepal
 

distance or altitude difference, capturing spatial and
altitudinal relationships.
3) Soil Dataset: Soil data for the 1359 locations was
collected via an API provided by [ 18]. This dataset [ 19]
includes Longitude, Latitude, Parent Soil, pH, Clay , Or-
ganic Matter, T otal Nitrogen, and P2O5. F eature selection
removed attributes less correlated with crop growth (e.g.,
sand, boron).
4) V egetable Dataset: V egetable data was curated from
sources like [ 20] and [ 21], detailing optimal growth con-
ditions. The dataset [ 22] [23] includes Name, and mini-
mum/maximum values for Soil pH, Phosphorus, Potas-
sium, Nitrogen, W eather T emperature, Precipitation, Hu-
midity , and Altitude. F eature selection focused on key
growth factors.
B. W eather Prediction
W eather prediction utilizes historical data from the 1359
locations over 40 years. Preprocessing involved handling
missing values (interpolation, forward-filling) and normal-
izing data. T emporal snapshots (weekly/monthly) were
created to capture spatio-temporal patterns.
T wo models were employed:
1) Spatio-temporal Graph Convolutional Network
(GCN): This model ( 2) uses a graph where nodes are loca-
tions. Edges represent geographical proximity or weather
similarity . GCN layers capture spatial dependencies by
propagating information between neighboring nodes. The
model is extended to process sequences of graphs, captur-
ing temporal trends.
2) T ransformer-Based Graph Model: The T ransformer-
Based Spatio-T emporal Graph Convolutional Network
predicts multi-step data over the network of nodes. It
combines temporal convolution (short-term patterns), a
temporal transformer block (long-term dependencies), and
graph convolution (spatial dependencies) before a final
prediction layer.
3) T raining and V alidation: Models were trained us-
ing supervised learning with an Adam optimizer and a
learning rate scheduler [ 24]. Data was split 70% training,
15% validation, and 15% testing. Hyperparameter tuning
and cross-validation were used, along with early stoppage
and dropout for regularization. Performance was evaluated
using Mean Absolute Error (MAE) and Root Mean Square
Error (RMSE).
C. Crop Recommendation
Crop recommendation integrates predicted weather
data with soil data and historical crop information ( 3).
The recommendation engine matches location-specific
environmental factors (predicted weather, soil data, al-
titude) with the optimal growth conditions listed in the
vegetable dataset. A suitability score is calculated for each
vegetable using an inverse distance metric:
score+ =1
1 +|location_data[feature] −mid_val |where mid_val is the midpoint of the vegetable’s ideal
range for a feature. Higher scores indicate better suit-
ability . The engine outputs a ranked list of recommended
crops for a given location.
IV. Experimental Setup
The experiments were conducted at the A.I. Lab,
Kathmandu University , located in Panchkhal, Kavre.
The experimental setup utilized a local PC with specific
hardware and software configurations. Despite facing chal-
lenges such as system shutdowns caused by overheating
or insuﬀicient memory resources, the experiments were
successfully completed. Remote access to the PC was
facilitated using an ngrok tunnel.
The hardware employed for training all models fea-
tured an 11th Gen Intel(R) Core(TM) i7-11700K CPU,
an NVIDIA GeF orce R TX 3080 Ti GPU with 12 GB
GDDR6X memory and 10240 CUDA Cores operating
at 1.67 GHz, and 16 GB (8GB x 2) of DIMM DDR4
Synchronous RAM running at 3200 MHz. The software
environment was based on Ubuntu 22.04.3 L TS with kernel
version 6.2.0-37-generic. Model implementations utilized
T orch-geometric, built upon PyT orch 1.9.0+cu102, with
Python 3.8.16 managed via Conda 23.1.0. The backend
system was developed using Django and SQLite3, while
the mobile application was built with the React Na-
tive framework. Pinecone was used as the cloud vector
database for storing RAG documents, and Cohere’s free
API plan was utilized for retrieval augmentation.
V. Results
A. W eather Prediction Model Performance
T o evaluate the prediction of 30-day weather
parameters—humidity , precipitation, and temperature—
we compared two models: a Spatio-T emporal Graph Neu-
ral Network (STGNN) and a transformer-based graph
model. The key metric used was Mean Squared Error
(MSE).
The STGNN achieved a lower MSE of 0.011 compared
to 0.013 for the transformer-based model, indicating better
performance in capturing spatial-temporal dependencies.
This advantage is likely due to the STGNN’s ability to
explicitly model interactions between geographical loca-
tions.
While the transformer-based model showed slightly
higher MSE, it still performed well in capturing temporal
patterns, thanks to its sequence modeling capabilities.
However, its lack of spatial structure may limit its perfor-
mance in spatially complex data.
B. Mobile Interface
The mobile application stands as a pivotal component
of the vegetable and fruit recommendation system, func-
tioning as the primary interface through which users,
specifically farmers, interact with the system. By offering
an intuitive and user-friendly platform, it successfully

       
    
Fig. 2. Architecture of Spatio temporal Graph Neural Networks.
   
       
   
Fig. 3. Flow Diagram of Crop Recommendation System (Model Inferencing)
   
Model Mean Squared Error (MSE)
Spatio-T emporal Graph Neural Network (STGNN) 0.011
T ransformer-Based Graph Model 0.013
T ABLE II
Comparison of Model Performance Based on MSE
bridges the gap between sophisticated backend algorithms
and the end-users, providing seamless, location-based rec-
ommendations directly at their fingertips. The application
is developed with support for the Nepali language, which
significantly enhances its usability and accessibility for
local farmers. Key features integrated into the application
include an easy-to-navigate interface, real-time updates
on crop recommendations tailored to the user’s location,
and functionalities that could potentially allow for cus-
tomization of recommendations based on various needs or
preferences.
Upon launching the application, it initially requests
location permission from the user. Once permission isgranted, the system communicates with the backend to
retrieve recommended crops specifically suited for the
farmer’s geographical location and current conditions.
The application then displays information about the
suitable crops, including details about their ideal growing
conditions derived from the dataset, alongside environ-
mental attributes predicted by our system, and general
descriptive information about the crops themselves. This
comprehensive presentation helps farmers understand the
rationale behind the recommendations. F urthermore, the
application extends its utility by providing information
not only on the recommended crops but also on other
vegetables and fruits that might be of interest to farm-

   
  
Fig. 4. MSE loss for the T ransformer Based Graph Model
 
   
  
Fig. 5. MSE loss for the STGCN
 
ers. Users can browse a list of these crops and access
detailed information about each one.This mobile platform
offers unparalleled accessibility , enabling farmers to ac-
cess valuable recommendations and information anytime,
anywhere, directly supporting more informed decisions in
their farming practices.
C. RAG System for F armer Queries
A Retrieval-Augmented Generation (RAG) system was
implemented as a crucial component to effectively ad-
dress queries posed by farmers. This system is designed
to combine the strengths of information retrieval withtext generation capabilities. By first retrieving relevant
information from a knowledge base and then utilizing a
generation model to formulate context-sensitive answers,
the system aims to provide clear, practical, and accurate
solutions to common agricultural questions encountered
by farmers.
The RAG system underwent testing with a diverse
range of queries submitted by farmers, covering topics that
included specific crop recommendations, methods for pest
control, advice related to soil health management, and in-
quiries about ongoing agricultural subsidies. Observations

from these tests indicated that the system consistently
generated responses that were both relevant to the query
and helpful to the farmer. The retrieval component played
a key role in ensuring that the generated answers were
well-grounded in accurate information obtained from the
knowledge base. Simultaneously , the generation model
effectively tailored the responses to directly address the
specific needs and contexts expressed in each individual
query . The system supports queries in multiple convenient
languages, including Nepali and English, and is configured
to provide the final answers in Nepali language, enhancing
accessibility for local users. The RAG system employs
a hybrid approach for document retrieval, combining
methods like SPLADE with dense and sparse vector
representations to improve the accuracy and eﬀiciency of
information retrieval from the knowledge base.
While quantitative metrics were not systematically
applied for evaluation, qualitative assessments based on
user interactions and feedback revealed that the RAG
system provides valuable assistance to farmers.
VI. Conclusion
This research developed a localized crop recommen-
dation system for Nepal, leveraging weather data from
1359 locations, predicted using Spatio-T emporal Graph
Neural Networks (STGNN) and T ransformer-based graph
models. Both models predicted 30-day weather parame-
ters based on 100 days of historical data. The STGNN
demonstrated superior performance with a Mean Squared
Error (MSE) of 0.011, compared to the T ransformer’s MSE
of 0.013, primarily due to its effectiveness in capturing
spatio-temporal dependencies crucial for regional weather
interactions. While the T ransformer exhibited strength in
long-term temporal patterns, the STGNN’s graph-based
approach proved more adept for this application.
The system integrates these weather predictions with
local soil attributes using a scoring algorithm to provide
tailored crop recommendations. Key contributions include
the collection and preparation of localized weather and
soil data for Nepal, the development and evaluation of
STGNN and T ransformer models for weather forecasting,
the integration of weather and soil data for recommenda-
tions, and the implementation of a chat system to address
farmer queries. This work provides a robust framework
and highlights the potential of advanced AI models in
agricultural planning.
VII. Limitations and F uture W ork
Despite its promise, the system has limitations. It relies
on potentially inconsistent historical data, lacks compre-
hensive quantitative evaluation metrics specifically for
crop recommendation accuracy , and the models may not
fully capture highly complex environmental interactions.
Localization challenges regarding microclimates and cul-
tural practices, and accessibility of necessary technology
for all farmers, also exist.F uture work should focus on expanding data sources
to include real-time satellite data and local reports, de-
veloping robust quantitative metrics for recommendation
performance, enhancing model complexity with hybrid ar-
chitectures, adapting recommendations for microclimates
and local factors, and improving user accessibility through
a mobile application with offline capabilities. Adapting
the framework for other regions with similar agricultural
contexts is also a valuable future direction.
References
[1] S. Pudumalar, E. Ramanujam, R. Harine Rajashree, C. Kavya,
T. Kiruthika, and J. Nisha. Crop recommendation system for
precision agriculture. In 2016 Eighth International Conference
on Advanced Computing (ICoAC), pages 32–36, 2017.
[2] F ranco Scarselli, Marco Gori, Ah Chung T soi, Markus Hagen-
buchner, and Gabriele Monfardini. The graph neural network
model. IEEE T ransactions on Neural Networks, 20(1):61–80,
2009.
[3] A V aswani. Attention is all you need. Advances in Neural
Information Processing Systems, 2017.
[4] P . Mehra et al. W eather prediction for crop yield using linear
regression. International Journal of Agricultural Research, 2017.
[5] S. Kumari et al. Climate-based crop prediction using machine
learning models. Journal of Agricultural Informatics, 2020.
[6] Y. Liu et al. T ransformer-based models for time-series forecast-
ing in agriculture. In Proceedings of the 2020 Conference on AI
in Climate, 2020.
[7] A. Chaudhari et al. Soil-based crop recommendation using
decision trees. Soil Science and Agriculture, 2018.
[8] P . Sharma et al. Hybrid machine learning model for crop
recommendation. In Agricultural Informatics Conference, 2018.
[9] S. Li et al. Spatio-temporal graph neural networks for yield
prediction. IEEE T ransactions on Neural Networks, 2021.
[10] Y. Xu et al. Spatio-temporal graph convolutional networks
for environmental modeling. In Proceedings of the Neural
Information Conference, 2022.
[11] J. Zhang et al. Graph neural networks for spatial prediction in
agriculture. Machine Learning in Agriculture, 2018.
[12] F. W ang et al. Enhancing crop recommendation with spatio-
temporal gnns. Journal of Agricultural Data Science, 2023.
[13] M. Lin et al. Crop yield prediction using transformers. AI in
Agriculture, 2021.
[14] Google earth, 2024. Accessed: 2024-11-27.
[15] Prazzwal07. Location dataset, 2024. Accessed: 2024-12-02.
[16] Prazzwal07. W eather dataset, 2024. Accessed: 2024-12-02.
[17] Nasa power project, 2024. Accessed: 2024-11-27.
[18] Soil information system, 2024. Accessed: 2024-11-27.
[19] Prazzwal07. Soil dataset, 2024. Accessed: 2024-12-02.
[20] National Agriculture Science and T echnology Institute. Com-
prehensive insights in vegetables, 2024. Accessed: 2024-11-27.
[21] NCFD. Dry fruit booklet final part-1, 2021. Accessed: 2024-11-
27.
[22] Prazzwal07. V egetable dataset, 2024. Accessed: 2024-12-02.
[23] Prazzwal07. F ruit dataset, 2024. Accessed: 2024-12-02.
[24] Diederik P . Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
Appendix
A. Detailed Data V ariables
Location Data V ariables: The variables collected for
each location include geographical coordinates and eleva-
tion, crucial for mapping the diverse agricultural landscape
of Nepal. The specific variables are detailed in T able III .
W eather Data V ariables: The comprehensive weather
dataset includes several key variables essential for fore-
casting and understanding their impact on crop growth.
These variables are listed and described in T able IV .

V ariable Description Unit Data Type
Longitude Geographical east-west coordinate Degrees Float
Latitude Geographical north-south coordinate Degrees Float
Altitude Elevation above sea level Meters Float
Location Name Name of the location N/A String
T ABLE III
Data variables collected for each location.
V ariable Description Unit Data Type
T emperature (T2M) A verage daily temperature at 2m height °C Float
Precipitation (PRECTOTCORR) Daily precipitation mm Float
Humidity (RH2M) Daily average relative humidity at 2m height % Float
W et Bulb T emperature (T2MWET) W et bulb temperature at 2m height °C Float
Wind Speed (WS10M) Wind speed at 10m height m/s Float
Earth Skin T emperature (TS) A verage daily earth surface temperature °C Float
T ABLE IV
W eather data variables used in the model.
Soil Data V ariables: The soil dataset provides essential
information on soil fertility and characteristics for each
location, used in the crop recommendation process. The
dataset contains the following attributes:
•Location: The name or identifier of the location where
the soil sample was collected.
•Longitude: The east-west geographic coordinate of
the sampling point.
•Latitude: The north-south geographic coordinate of
the sampling point.
•Parent Soil: The type of parent material or soil
classification (e.g., alluvial, loamy).
•pH: The pH level of the soil, indicating its acidity or
alkalinity .
•Clay: The percentage of clay particles in the soil.
•Organic Matter: The percentage of organic material
in the soil, contributing to its fertility .
•T otal Nitrogen: The total nitrogen content of the soil,
a key nutrient for plant growth.
•P2O5: The phosphorus content measured as available
phosphate (P2O5), crucial for root development.
V egetable Dataset V ariables: The vegetable dataset
outlines the optimal growth conditions for various crops,
enabling tailored recommendations based on environmen-
tal and soil factors. The dataset includes the following
attributes:
•Name: The name of the vegetable.
•Soil pH Min: The minimum soil pH level suitable for
the vegetable’s growth.
•Soil pH Max: The maximum soil pH level suitable for
the vegetable’s growth.
•Soil Phosphorus Min: The minimum soil phosphorus
level (as P2O5) required.
•Soil Phosphorus Max: The maximum phosphorus
level (as P2O5) suitable.
•Soil Potassium Min: The minimum potassium level
required (as K2O).
•Soil Potassium Max: The maximum potassium levelsuitable (as K2O).
•Soil Nitrogen Min: The minimum nitrogen level re-
quired.
•Soil Nitrogen Max: The maximum nitrogen level
suitable.
•W eather T emperature Min: The minimum tempera-
ture for optimal growth.
•W eather T emperature Max: The maximum tempera-
ture suitable for the vegetable.
•W eather Precipitation Min: The minimum precipita-
tion (mm) required.
•W eather Precipitation Max: The maximum precipi-
tation the vegetable can tolerate.
•W eather Humidity Min: The minimum humidity level
required.
•W eather Humidity Max: The maximum humidity
level suitable.
•Altitude Min: The minimum altitude (in meters)
where the vegetable can grow.
•Altitude Max: The maximum altitude where the
vegetable can grow.
•Matching Score: A score representing how well the
vegetable matches environmental and soil conditions.
B. Dataset A vailability
The datasets utilized in this research are made available
for further study and reproducibility . The location data
is available at [ 15]. The weather data, originally sourced
from NASA Power [ 17], is provided at [ 16]. The soil data,
obtained through the NARC API [ 18], can be accessed at
[19]. The vegetable and fruit datasets, specifically curated
for optimal crop growth conditions, are available at [ 22]
and [ 23], respectively .
C. Mobile Application Screenshots
Screenshots illustrating additional details of the mobile
application interface and its features are provided below,
complementing the main description in the Results sec-
tion.

Fig. 6. Details of recommended crop
 Fig. 7. List of vegetables

Fig. 8. RAG implementation screenshot 1