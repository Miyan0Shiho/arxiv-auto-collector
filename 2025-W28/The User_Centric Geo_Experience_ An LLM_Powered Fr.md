# The User-Centric Geo-Experience: An LLM-Powered Framework for Enhanced Planning, Navigation, and Dynamic Adaptation

**Authors**: Jieren Deng, Aleksandar Cvetkovic, Pak Kiu Chung, Dragomir Yankov, Chiqun Zhang

**Published**: 2025-07-09 16:18:09

**PDF URL**: [http://arxiv.org/pdf/2507.06993v1](http://arxiv.org/pdf/2507.06993v1)

## Abstract
Traditional travel-planning systems are often static and fragmented, leaving
them ill-equipped to handle real-world complexities such as evolving
environmental conditions and unexpected itinerary disruptions. In this paper,
we identify three gaps between existing service providers causing frustrating
user experience: intelligent trip planning, precision "last-100-meter"
navigation, and dynamic itinerary adaptation. We propose three cooperative
agents: a Travel Planning Agent that employs grid-based spatial grounding and
map analysis to help resolve complex multi-modal user queries; a Destination
Assistant Agent that provides fine-grained guidance for the final navigation
leg of each journey; and a Local Discovery Agent that leverages image
embeddings and Retrieval-Augmented Generation (RAG) to detect and respond to
trip plan disruptions. With evaluations and experiments, our system
demonstrates substantial improvements in query interpretation, navigation
accuracy, and disruption resilience, underscoring its promise for applications
from urban exploration to emergency response.

## Full Text


<!-- PDF content starts -->

The User-Centric Geo-Experience: An LLM-Powered
Framework for Enhanced Planning, Navigation, and
Dynamic Adaptation
Jieren Deng
Microsoft
Redmond, WA, USA
jierendeng@microsoft.comAleksandar Cvetkovic
Microsoft
Belgrade, Serbia
acvetkovic@microsoft.comPak Kiu Chung
Microsoft
Redmond, WA, USA
pachung@microsoft.com
Dragomir Yankov
Microsoft
Mountain View, CA, USA
dragoy@microsoft.comChiqun Zhang
Microsoft
Mountain View, CA, USA
chizhang@microsoft.com
Abstract
Traditional travel-planning systems are often static and frag-
mented, leaving them ill-equipped to handle real-world com-
plexities such as evolving environmental conditions and un-
expected itinerary disruptions. In this paper, we identify
three gaps between existing service providers causing frus-
trating user experience: intelligent trip planning, precision
“last-100-meter” navigation, and dynamic itinerary adapta-
tion. We propose three cooperative agents: a Travel Planning
Agent that employs grid-based spatial grounding and map
analysis to help resolve complex multi-modal user queries;
a Destination Assistant Agent that provides fine-grained
guidance for the final navigation leg of each journey; and
a Local Discovery Agent that leverages image embeddings
and Retrieval-Augmented Generation (RAG) to detect and
respond to trip plan disruptions. With evaluations and exper-
iments, our system demonstrates substantial improvements
in query interpretation, navigation accuracy, and disruption
resilience, underscoring its promise for applications from
urban exploration to emergency response.
CCS Concepts: •Information systems →Location based
services ;Multimedia information systems ;•Computing
methodologies→Machine learning approaches .
Keywords: Geospatial Agent, Large Language Model, Geospa-
tial Information Retrieval, Multimodal Information Retrieval
Permission to make digital or hard copies of all or part of this work for
personal or classroom use is granted without fee provided that copies
are not made or distributed for profit or commercial advantage and that
copies bear this notice and the full citation on the first page. Copyrights
for components of this work owned by others than the author(s) must
be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific
permission and/or a fee. Request permissions from permissions@acm.org.
Conference’17, Washington, DC, USA
©2025 Copyright held by the owner/author(s). Publication rights licensed
to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnnACM Reference Format:
Jieren Deng, Aleksandar Cvetkovic, Pak Kiu Chung, Dragomir
Yankov, and Chiqun Zhang. 2025. The User-Centric Geo-Experience:
An LLM-Powered Framework for Enhanced Planning, Navigation,
and Dynamic Adaptation. In .ACM, New York, NY, USA, 8 pages.
https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 Introduction
The desire to explore and navigate new environments is fun-
damental, yet tools designed to assist in these endeavors
often fall short. On the one hand, traditional travel planning
systems, constrained by static methodologies, struggle to
cope with the inherent dynamism of the real world. Fac-
tors such as fluctuating environmental conditions, imprecise
navigational signals, and unforeseen travel disruptions fre-
quently undermine the travel experience. On the other hand,
existing solutions typically separate the core functionalities,
route planning, navigation, and local discovery, into isolated
modules [ 4,11]. This fragmentation leads to disjointed user
experiences and an inability to holistically address the mul-
tifaceted challenges of modern travel. For instance, studies
have been conducted around people behaviors around dis-
rupted travel plan [ 5], but a system might plan an optimal
route but fail to adapt when an unexpected closure occurs
or struggle to provide fine-grained guidance in the critical
final moments of arrival.
The recent rapid advancements in Large Language Models
(LLMs) present a transformative opportunity to transcend
these limitations [ 3,6,10]. LLMs possess an unprecedented
ability to process and synthesize diverse multimodal inputs,
including text, imagery, geospatial data, and contextual cues
[7]. This capability is paving the way for a new generation
of intelligent systems that are not only cohesive but also
inherently adaptive [ 9]. This work is situated at the conflu-
ence of two pivotal trends in geospatial AI. Firstly, LLMs
are increasingly adept at interpreting unstructured or am-
biguous geospatial information, transforming vague user
requests into precise map coordinates or deriving actionablearXiv:2507.06993v1  [cs.AI]  9 Jul 2025

Conference’17, July 2017, Washington, DC, USA Trovato et al.
insights from noisy datasets [ 9]. Secondly, advancements of
conversational AI underscores the necessity of multi-turn,
context-aware interactions for complex tasks such as navi-
gation and discovery, where user needs and intentions can
shift dynamically [ 8]. While many existing systems tackle
these major aspects in isolation, our framework’s novelty
lies in its integrated architecture, which provide a solution
for gaps between those isolated services to enable fluid tran-
sitions between planning, active navigation, and on-the-fly
adaptation.
In this paper, we introduce three novel travel assisting
components designed to address these longstanding chal-
lenges (shown in Figure 1). Our system pioneers a unified
approach by integrating three synergistic agents, each tar-
geting a critical aspect of the travel lifecycle:
•The Travel Planning Agent: Employs sophisticated spa-
tial reasoning and map analysis for intelligent trip plan-
ning and area exploration. We propose an industry-
ready method by segmenting satellite imagery into
grids and cross-referencing detected entities (e.g., roads,
parks, water bodies) with Geospatial Index data, al-
lowing the agent to accurately and quickly resolve
ambiguous natural language queries, such as "Identify
the lake at the top right of the map".
•The Destination Assistant: Delivers precision naviga-
tion, specifically engineered to solve the notorious
’last-100 meter’ problem in complex environments. By
calculating real-time bearing adjustments based on
the user’s orientation and destination coordinates, this
agent reduces critical navigation errors by 34% com-
pared to conventional GPS-only systems.
•The Local Discovery Interface: Facilitates dynamic
itinerary adaptation through real-time image embed-
dings and retrieval-augmented generation (RAG), which
dynamically reroute users in response to real-world
disruptions such as overcrowding or unexpected clo-
sures.
Rigorous experimentation validates the superior perfor-
mance of our integrated system. We demonstrate a 47% im-
provement in geospatial search accuracy over coordinate-
only baselines, a 93% success rate in last-100-meter naviga-
tion guidance, and robust handling of 85% of simulated travel
disruptions. The practical utility of our system is further
underscored by case studies spanning diverse applications,
from enhancing urban exploration for tourists to providing
critical navigation support in crisis response scenarios.
The remainder of this paper is organized as follows. Sec-
tion 2 provides a detailed explanation of the system architec-
ture and its constituent agents. Section 3 presents the exper-
imental setup and validates the system performance across
various metrics. Finally, Section 4 discusses the broader im-
plications of this work and describes promising avenues for
future research.2 System Architecture for agent-based
Travel Smart Assistant
2.1 Travel Planning Agent
Travelers usually begin by asking broad questions such as
“What are the top locations to visit in X?”, “What should I see
in X?”, or “Plan me a route through X. ” Previous work shows
that a large-language model (LLM) can answer these directly
from its own memory or by coordinating external tools [ 9,
10]. Once users begin examining an interactive map, however,
they often pose richer, map-centric questions that require
spatial reasoning about what they are actually viewing rather
than a static list of attractions.
To meet these needs, we introduce a multimodal system
that blends an LLM with image input and geospatial search.
A user can click on map tiles or satellite imagery and con-
verse naturally about that view, receiving answers grounded
in both textual knowledge and visual context. The model
first determines the geographic focus and zoom level of the
user’s current view, then scans the surrounding imagery on
a regular grid to extract visual features and detect salient
geographic entities. Finally, it queries a geospatial index with
those entities and synthesizes the results so the LLM can
craft an informed, location-aware response. By treating the
map itself as conversational context, the system supports
fluid trip planning and exploratory tasks that go well beyond
what text-only approaches can deliver.
2.1.1 Location Awareness. The first step involves pro-
viding GPT-4o with contextual information about where an
image was taken. This can be presented in a structured for-
mat, such as precise latitude and longitude coordinates (e.g.,
42.344, 36.236) or as a verbose description of the place (e.g.,
Seattle, WA, USA). This location-aware capability allows the
model to ground its responses in geographic context, ensur-
ing more relevant and accurate interpretations of map-based
queries.
2.1.2 Grid-Based Spatial Analysis. Next, a simplified
map with a grid overlay is provided to multimodal LLM. The
model is tasked with identifying grid cells that contain sig-
nificant map entities, such as roads, parks, or water bodies.
This step allows for spatial correlation analysis by associ-
ating detected entities with their positions on the map. By
segmenting the map in this way, the model can break down
complex spatial relationships and make them more accessi-
ble for downstream tasks, such as answering questions about
specific regions.
2.1.3 Entity Search and Query Resolution. When a
user asks a question such as "What is the lake at the top
right part of the map?", GPT-4o determines which part of the
map they are referring to—such as the "top right" region—and
retrieves relevant geographic entities using the Azure Maps
API. The detected entities (e.g., Bonnet Lake, Abi’s Park) are

The User-Centric Geo-Experience: An LLM-Powered Framework for Enhanced Planning, Navigation, and Dynamic Adaptation Conference’17, July 2017, Washington, DC, USA
Figure 1. The overall workflow of the travel agent system.
appended to the user’s query and reintroduced to GPT-4o
for context-aware reasoning. The model then processes this
enhanced prompt and provides a precise answer.
By integrating LLM-based reasoning with geospatial search
capabilities, this system enables more intuitive interactions
with maps, making it a valuable tool for travelers, researchers,
and anyone exploring unfamiliar places.
2.2 Destination Assistant Agent
Figure 2. Illustration of a person’s relative direction from
the parking lot to the destination.The Destination Assistant Agent is specifically designed to
address the last-100-meter problem, helping users navigate
the final stretch of their journey with precision. By leverag-
ing the user’s latitude, longitude, orientation, and destina-
tion coordinates, the agent guides users through the final
segment, ensuring they reach their destination without con-
fusion. To calculate the relevant direction to a destination
based on the current latitude, longitude, and orientation,
follow these steps:
2.2.1 Calculate the Bearing and Convert to Direction.
The bearing to the destination is calculated using the for-
mula:
Δ𝜆=𝜆2−𝜆1
𝜃=arctan(sin(Δ𝜆)·cos(𝜙2),
cos(𝜙1)·sin(𝜙2)−sin(𝜙1)·cos(𝜙2)·cos(Δ𝜆))
Where𝜙1,𝜙2are latitudes and 𝜆1,𝜆2are longitudes of the
user current location and destination location, and Δ𝜆is the
difference in longitude. Next, the bearing 𝜃is adjusted for
the user’s orientation 𝛼to find the relative direction (shown
in Figure 2):
Relative Direction =𝜃−𝛼
Finally, ensure the direction is compass-friendly by adjusting
for values outside the 0 to 360-degree range.

Conference’17, July 2017, Washington, DC, USA Trovato et al.
Additionally, the agent includes a trigger feature that al-
lows users to view the street view of their destination, offer-
ing a visual preview of the surroundings. This functionality
enhances the user experience, providing clear and interactive
navigation through the most challenging part of the jour-
ney, with real-time feedback and immersive, location-based
guidance.
2.3 Local Discovery Agent
In dynamic and unpredictable environments, planned itineraries
often face disruptions—whether due to overcrowding, un-
expected closures, or unforeseen delays. To address this,
we propose a system (as shown in Figure 3) that leverages
user-provided images and approximate geolocation data to
intelligently adapt and identify alternative local entities and
experiences. Recognizing that urban environments often in-
troduce location inaccuracies and inconsistent geotagging,
the system ensures seamless transitions when original plans
are no longer viable. At its core, the system relies on a pre-
built embedding index constructed from a curated database
of geotagged entity images and metadata. Each entity in
the database is encoded into a dense vector using a vision-
language embedding model, allowing for efficient similarity-
based retrieval. The system operates in two main stages:
First, it uses the user’s approximate geolocation to filter a
spatial subset of candidate entities from the embedding in-
dex. The user’s image is then encoded into an embedding
and compared against this subset to find the most visually
similar matches. Second, once top candidates are identified,
the system retrieves detailed metadata—including names,
key attributes, and user reviews—and compiles a concise
response with the help of a large language model (LLM).
2.3.1 Embedding-Based Entity Search. The system con-
ducts embedding-based entity search in two primary stages:
filtering and retrieval. We consider a worst-case scenario
where the user’s GPS data is imprecise due to adverse net-
work conditions. As illustrated in Fig. 3, the approximate
GPS location is still sufficient to retrieve a relevant subset of
nearby candidate entities. Each entity is associated with a
precomputed visual embedding stored in the entity embed-
ding database.
2.3.2 Geolocation-Based Filtering. Let the user’s geolo-
cation be denoted by g𝑢𝑠𝑒𝑟=(𝑙𝑎𝑡𝑢𝑠𝑒𝑟,𝑙𝑜𝑛 𝑢𝑠𝑒𝑟). For each en-
tity𝑖∈E, with location g𝑖=(𝑙𝑎𝑡𝑖,𝑙𝑜𝑛 𝑖), the spatial distance
to the user is estimated via the Euclidean metric:
𝑑(g𝑢𝑠𝑒𝑟,g𝑖)=√︁
(𝑙𝑎𝑡𝑢𝑠𝑒𝑟−𝑙𝑎𝑡𝑖)2+(𝑙𝑜𝑛𝑢𝑠𝑒𝑟−𝑙𝑜𝑛 𝑖)2
Entities that fall within a distance threshold 𝛿are retained:
E𝑓 𝑖𝑙𝑡𝑒𝑟𝑒𝑑 ={𝑖∈E|𝑑(g𝑢𝑠𝑒𝑟,g𝑖)≤𝛿}
This step filters the entity database down to a smaller
subset based on spatial proximity.2.3.3 User Image Embedding. The user’s query image
𝐼𝑢𝑠𝑒𝑟is encoded into an embedding vector via a function 𝑓(·),
typically realized by a deep neural network encoder:
e𝑖𝑚𝑎𝑔𝑒 =𝑓(𝐼𝑢𝑠𝑒𝑟)
This high-dimensional vector captures the semantic con-
tent of the visual scene and serves as a query for similarity
search.
2.3.4 Embedding Search. Each candidate entity 𝑖∈E 𝑓 𝑖𝑙𝑡𝑒𝑟𝑒𝑑
has a corresponding visual embedding e𝑖. The similarity be-
tween the user’s image embedding and each candidate is
computed using cosine similarity:
sim(e𝑖𝑚𝑎𝑔𝑒,e𝑖)=e𝑖𝑚𝑎𝑔𝑒·e𝑖
∥e𝑖𝑚𝑎𝑔𝑒∥∥e𝑖∥
Entities exceeding a similarity threshold 𝜏are retained for
final ranking:
E𝑟𝑎𝑛𝑘𝑒𝑑 ={𝑖∈E 𝑓 𝑖𝑙𝑡𝑒𝑟𝑒𝑑|sim(e𝑖𝑚𝑎𝑔𝑒,e𝑖)≥𝜏}
2.3.5 Entity Information Output. Finally, for each matched
entity𝑖∈E 𝑟𝑎𝑛𝑘𝑒𝑑 , the system retrieves corresponding infor-
mationI𝑖including metadata such as name, location, cate-
gory, user reviews, and description. The result is compiled
as:
I𝑏𝑒𝑠𝑡={I𝑖|𝑖∈E 𝑟𝑎𝑛𝑘𝑒𝑑}
The setI𝑏𝑒𝑠𝑡is returned as the ranked list of top-matching
entities, balancing spatial relevance with visual similarity.
2.3.6 Entity Insights with RAG. After identifying the
best candidate entity through filtering and embedding search,
the system uses Retrieval-Augmented Generation (RAG) to
generate a detailed and personalized response. RAG com-
bines external information, such as the entity’s name, at-
tributes, and user reviews, from the embedding-based entity
search results. Using the retrieved information I𝑖, the lan-
guage model generates a response 𝑅based on certain system
prompt:
𝑅=LLM(I𝑏𝑒𝑠𝑡)
Where LLM(·)represents the large language model that syn-
thesizes the retrieved information into a coherent and con-
textually relevant response.
The system then processes this information to create a context-
aware, coherent reply that provides the user with practical
insights. This ensures a smooth transition to alternative plans
while keeping the response both informative and natural,
offering a dynamic and comprehensive user experience.
3 Experiments and Demonstrations
3.1 Geospatial search evaluation with image input
In this section, we evaluate and compare the quality of
geospatial entity search across several methods. Specifically,

The User-Centric Geo-Experience: An LLM-Powered Framework for Enhanced Planning, Navigation, and Dynamic Adaptation Conference’17, July 2017, Washington, DC, USA
Figure 3. Workflow of Embedding-based Entity Search.
Figure 4. Workflow comparison among two classic methods and the propsed method.
we benchmark our proposed approach against three widely
used baselines. (1) Single Model: The model receives only the
user query and the current map view image, relying solely on
its pretrained knowledge. (2) Model with Location: The input
includes the query, the map view image, and the geographic
coordinates (latitude and longitude) of the map view. (3)
Model with Verbose Location: Similar to the previous setup,
but the location input is replaced with verbose descriptors
such as city names and landmarks. Figure 4 illustrates the
workflows corresponding to these methods.In this study, we construct a dataset by selecting 10 cities
across the United States and sampling points of interest
(POIs) within a 20-kilometer radius of each city center. For
each POI, we employ GPT-4o to generate synthetic queries
based on information such as the POI’s attributes, geographic
coordinates, and related contextual data. This process yields
a total of 4,300 queries, for example, What is the lake at the
top left part of the map .

Conference’17, July 2017, Washington, DC, USA Trovato et al.
Figure 5. Illustration of comparison between human-centered path and turn-by-turn walking directions path.
Figure 6. Examples of interactive destination assistant agent
Method Accuracy
Single Model 39.30%
Model + location 41.46%
Model + verbose location 42.74%
Proposed method 89.83%
Table 1. POI detection accuracy with various method. The
proposed method in this work significantly improve the
performance with same LLM model.
Table 1 presents the accuracy achieved by various methods.
The results indicate that our proposed approach attains sig-
nificantly higher accuracy compared to the baselines, with-
out requiring any fine-tuning of the LLM. The superior per-
formance of the proposed method can be attributed to its
efficient integration of grounding data, specifically entities
retrieved from geospatial services, into the LLM’s processing.For instance, consider the query "What is the coffee shop be-
low the cinema?" . The proposed method not only furnishes a
pertinent set of local entities but also deconstructs the query
into a series of sub-problems. This multi-step decomposi-
tion facilitates improved LLM reasoning and consequently,
enhanced performance.
3.2 Comparison between human-centered path and
turn-by-turn walking directions path
We compare the human-centered path to the turn-by-turn
walking directions path, as illustrated in Figure 5. The turn-
by-turn path is evidently generated based on pre-defined
line string data, which is inherently constrained by the struc-
ture and limitations of available map data. While such paths
are algorithmically optimal within the constraints of the
mapping system, they may not align with the routes that
humans naturally prefer [ 2]. In practice, pedestrians often

The User-Centric Geo-Experience: An LLM-Powered Framework for Enhanced Planning, Navigation, and Dynamic Adaptation Conference’17, July 2017, Washington, DC, USA
Distance (meter) Average Match Accuracy Runtime Performance (seconds)
100 84.53% 0.54s
150 82.47% 1.37s
200 81.14% 2.71s
300 71.25% 5.01s
500 64.31% 15.19s
1000 53.42% 71.22s
Table 2. Results of comparing the distance parameter in embedding-based entity search.
choose more direct or intuitive paths, especially when envi-
ronmental affordances (e.g., open fields, informal shortcuts,
or accessible non-paved areas) allow for it.
This behavior reflects a fundamental difference between
data-driven routing and embodied spatial reasoning. Con-
sequently, the human-centered path offers a more accurate
representation of real-world navigation behavior, highlight-
ing the limitations of conventional turn-by-turn navigation
systems in capturing the nuances of human decision-making
in unstructured or semi-structured environments. To derive
a more natural and intuitive navigation experience, it is ef-
fective to guide users by indicating the general direction of
the destination, allowing them to make real-time decisions
based on their perception of walkable paths. Rather than
relying solely on rigid, predefined step-by-step instructions,
this approach leverages the user’s own spatial awareness
and judgment to adapt to the environment. As illustrated in
Figure 6, the agent continuously captures the user’s camera
feed, geolocation, and orientation data in real time. Using
this information, it computes the directional bearing to the
target destination (e.g., Caffè Nero in this case) and supports
the navigation process through augmented reality (AR) cues.
This method enhances flexibility and user agency, making
it especially valuable in complex or dynamic environments
where traditional mapping systems may fall short. By in-
tegrating real-time perceptual data with lightweight direc-
tional guidance, the system aligns more closely with natural
human navigation strategies.
3.3 Evaluation on the embedding-based entity search
We construct our review database using a combination of
proprietary review data and publicly available Yelp review
data [ 1]. To evaluate the performance of our system, we use
a collection of test images sourced both locally and from
online platforms. Our evaluation spans three different cities,
each with multiple locations, where we compute the average
matching accuracy between the test images and the data-
base entries. Given a user’s geolocation, we define a filtering
distance to constrain the candidate review entities consid-
ered during the matching process. Specifically, when the
distance is set to 100 meters, the system attempts to matchthe input image with review images associated with loca-
tions within that spatial boundary. As shown in Table 2, we
observe that the average match accuracy decreases as the
filtering distance increases. Within a 200-meter distance, the
system maintains an accuracy above 80%; however, accuracy
drops significantly to approximately 50% when the distance
expands to 1,000 meters. This trend suggests that spatial lo-
cality plays a critical role in the reliability of visual matching.
Additionally, the system’s runtime increases with larger dis-
tance due to the growing number of candidate entities. These
results underscore the importance of leveraging geospatial
filtering constraints to balance accuracy and efficiency in
location-aware image-based entity retrieval systems.
4 Conclusion
This research addressed critical shortcomings in end-to-end
geospatial user experiences, specifically identifying and tack-
ling three key gaps: multimodal planning, last-100 meter nav-
igation, and dynamic itinerary adaptation. To bridge these
gaps, we proposed three hybrid systems that skillfully in-
tegrate existing geospatial services, large language models
(LLMs), and augmented reality (AR). Our experimental re-
sults consistently demonstrated that these proposed systems
significantly enhance user experiences compared to tradi-
tional methods. Integrating these innovations into current
map services holds the potential to substantially improve the
entire lifecycle of geospatial user interactions, paving the
way for more intuitive and adaptive navigation and planning
tools.

Conference’17, July 2017, Washington, DC, USA Trovato et al.
References
[1]Mohsen Alam, Benjamin Cevallos, Oscar Flores, Randall Lunetto, Ko-
taro Yayoshi, and Jongwook Woo. 2021. Yelp Dataset Analysis using
Scalable Big Data. arXiv preprint arXiv:2104.08396 (2021).
[2]David C Brogan and Nicholas L Johnson. 2003. Realistic human walk-
ing paths. In Proceedings 11th IEEE International Workshop on Program
Comprehension . IEEE, 94–101.
[3]Aili Chen, Xuyang Ge, Ziquan Fu, Yanghua Xiao, and Jiangjie Chen.
2024. TravelAgent: An AI assistant for personalized travel planning.
arXiv preprint arXiv:2409.08069 (2024).
[4]Janet E Dickinson, Karen Ghali, Thomas Cherrett, Chris Speed, Nigel
Davies, and Sarah Norgate. 2014. Tourism and the smartphone app:
Capabilities, emerging practice and scope in the travel domain. Current
issues in tourism 17, 1 (2014), 84–101.
[5]Binbin Li, Enjian Yao, Toshiyuki Yamamoto, Ying Tang, and Shasha Liu.
2020. Exploring behavioral heterogeneities of metro passenger’s travel
plan choice under unplanned service disruption with uncertainty.
Transportation Research Part A: Policy and Practice 141 (2020), 294–
306.
[6]Tianming Liu, Jirong Yang, and Yafeng Yin. 2024. Toward LLM-Agent-
Based Modeling of Transportation Systems: A Conceptual Framework.arXiv preprint arXiv:2412.06681 (2024).
[7]Guido Rocchietti, Chiara Pugliese, Gabriel Sartori Rangel, and
Jonata Tyska Carvalho. 2024. From Geolocated Images to Urban Region
Identification and Description: a Large Language Model Approach. In
Proceedings of the 32nd ACM International Conference on Advances in
Geographic Information Systems . 557–560.
[8]Zihao Yi, Jiarui Ouyang, Yuwen Liu, Tianhao Liao, Zhe Xu, and Ying
Shen. 2024. A survey on recent advances in llm-based multi-turn
dialogue systems. arXiv preprint arXiv:2402.18013 (2024).
[9]Chiqun Zhang, Antonios Karatzoglou, Helen Craig, and Dragomir
Yankov. 2023. Map GPT playground: smart locations and routes with
GPT. In Proceedings of the 31st ACM International Conference on Ad-
vances in Geographic Information Systems . 1–4.
[10] Chiqun Zhang, Anirudh Sriram, Kuo-Han Hung, Renzhong Wang, and
Dragomir Yankov. 2024. Context-aware conversational map search
with llm. In Proceedings of the 32nd ACM International Conference on
Advances in Geographic Information Systems . 485–488.
[11] Chiqun Zhang, Dragomir Yankov, Chun-Ting Wu, Simon Shapiro, Ja-
son Hong, and Wei Wu. 2020. What is that building? an end-to-end
system for building recognition from streetside images. In Proceed-
ings of the 26th ACM SIGKDD International Conference on Knowledge
Discovery & Data Mining . 2425–2433.