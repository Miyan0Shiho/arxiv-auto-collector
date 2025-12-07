# AskNearby: An LLM-Based Application for Neighborhood Information Retrieval and Personalized Cognitive-Map Recommendations

**Authors**: Luyao Niu, Zhicheng Deng, Boyang Li, Nuoxian Huang, Ruiqi Liu, Wenjia Zhang

**Published**: 2025-12-02 07:47:31

**PDF URL**: [https://arxiv.org/pdf/2512.02502v1](https://arxiv.org/pdf/2512.02502v1)

## Abstract
The "15-minute city" envisions neighborhoods where residents can meet daily needs via a short walk or bike ride. Realizing this vision requires not only physical proximity but also efficient and reliable access to information about nearby places, services, and events. Existing location-based systems, however, focus mainly on city-level tasks and neglect the spatial, temporal, and cognitive factors that shape localized decision-making. We conceptualize this gap as the Local Life Information Accessibility (LLIA) problem and introduce AskNearby, an AI-driven community application that unifies retrieval and recommendation within the 15-minute life circle. AskNearby integrates (i) a three-layer Retrieval-Augmented Generation (RAG) pipeline that synergizes graph-based, semantic-vector, and geographic retrieval with (ii) a cognitive-map model that encodes each user's neighborhood familiarity and preferences. Experiments on real-world community datasets demonstrate that AskNearby significantly outperforms LLM-based and map-based baselines in retrieval accuracy and recommendation quality, achieving robust performance in spatiotemporal grounding and cognitive-aware ranking. Real-world deployments further validate its effectiveness. By addressing the LLIA challenge, AskNearby empowers residents to more effectively discover local resources, plan daily activities, and engage in community life.

## Full Text


<!-- PDF content starts -->

AskNearby: An LLM-Based Application for Neighborhood
Information Retrieval and Personalized Cognitive-Map
Recommendations
Luyao Niu∗
Zhicheng Deng∗
Peking University
Qianmo Smart Link
Shenzhen, Guangdong, China
luyao0160@stu.pku.edu.cn
zhicheng.deng@stu.pku.edu.cnBoyang Li†
New York University
Brooklyn, NY, USA
boyang.li@nyu.eduNuoxian Huang
Imperial College London
London, United Kingdom
n.huang25@imperial.ac.uk
Ruiqi Liu
Tencent
Shenzhen, Guangdong, China
ruiqiliusysu@gmail.comWenjia Zhang
Tongji University
Shanghai, China
wenjiazhang@tongji.edu.cn
Abstract
The "15-minute city" envisions neighborhoods where residents can
meet daily needs via a short walk or bike ride. Realizing this vision
requires not only physical proximity but also efficient and reliable
access to information about nearby places, services, and events. Ex-
isting location-based systems, however, focus mainly on city-level
tasks and neglect the spatial, temporal, and cognitive factors that
shape localized decision-making. We conceptualize this gap as the
Local Life Information Accessibility (LLIA) problem and introduce
AskNearby, an AI-driven community application that unifies re-
trieval and recommendation within the 15-minute life circle.AskN-
earbyintegrates (i) a three-layer Retrieval-Augmented Generation
(RAG) pipeline that synergizes graph-based, semantic-vector, and
geographic retrieval with (ii) a cognitive-map model that encodes
each user’s neighborhood familiarity and preferences. Experiments
on real-world community datasets demonstrate thatAskNearby
significantly outperforms LLM-based and map-based baselines in
retrieval accuracy and recommendation quality, achieving robust
performance in spatiotemporal grounding and cognitive-aware
ranking. Real-world deployments further validate its effectiveness.
By addressing the LLIA challenge,AskNearbyempowers residents
to more effectively discover local resources, plan daily activities,
and engage in community life.
∗Both authors contributed equally to this research.
†Corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
GeoAI ’25, Minneapolis, MN, USA
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-2179-3/2025/11
https://doi.org/10.1145/3764912.3770813CCS Concepts
•Information systems →Location based services;Informa-
tion retrieval;•Computing methodologies →Natural language
processing;Planning and scheduling.
Keywords
Large Language Models, Retrieval-Augmented Generation, Spa-
tiotemporal Knowledge Graph, Local Information, 15-Minute City
ACM Reference Format:
Luyao Niu, Zhicheng Deng, Boyang Li, Nuoxian Huang, Ruiqi Liu, and Wen-
jia Zhang. 2025. AskNearby: An LLM-Based Application for Neighborhood
Information Retrieval and Personalized Cognitive-Map Recommendations.
InThe 8th ACM SIGSPATIAL International Workshop on AI for Geographic
Knowledge Discovery (GeoAI ’25), November 3–6, 2025, Minneapolis, MN, USA.
ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3764912.3770813
1 Introduction
The 15-minute city concept envisions that people can fulfill ba-
sic daily needs—such as grocery shopping, healthcare, education,
and leisure—within a short walking or cycling distance from their
homes, typically within 15 minutes [ 17]. This idea has gained wide-
spread attention among urban planners and policymakers, who
increasingly emphasize proximity-based planning and active trans-
portation as essential principles for promoting sustainable urban
development. In parallel with this, researchers have conducted em-
pirical studies on whether residents can reach nearby destinations
and access urban functions [1, 4, 12, 24].
Beyond physical accessibility, digitalization represents another
core principle of the 15-minute city. It entails leveraging smart
city technologies—such as big data and the Internet of Things—to
support real-time information delivery, citizen engagement, and
efficient urban resource management [ 11]. However, digital infras-
tructure alone does not guarantee that individuals can effectively
access or utilize localized information [ 5]. This raises a critical but
largely unaddressed question: Can people access relevant informa-
tion about nearby places and services to make informed decisions inarXiv:2512.02502v1  [cs.IR]  2 Dec 2025

GeoAI ’25, November 3–6, 2025, Minneapolis, MN, USA Niu et al.
their daily lives, especially in dynamic environments where places,
facilities, and human activities are constantly changing?
Existing applications are ill-equipped to address this need of
accessing local information at a granular, neighborhood scale [ 3,
16]. Their failure stems from an inability to model the nuanced
spatiotemporal context, ensuring that the offered information is
both timely and locally relevant. To overcome this limitation, it is
necessary to support two complementary processes through which
users interact with local information:active retrievalandpassive
recommendation.
In active retrieval, users explicitly seek information regarding
nearby services or events, often formulating queries inherently
containing spatiotemporal constraints (e.g., ’Which supermarket
near me is open late?’). Traditional search engines rely heavily on
keyword-based matching and fixed boundaries, such as a prede-
fined search radius or time windows, failing to grasp the complex
spatial and semantic intent in human queries [ 34]. Recent studies
have explored the potential of Large Language Models (LLMs) to
interpret semantic and spatiotemporal constraints on local informa-
tion and activities, but most approaches still treat space and time as
isolated filters rather than as interdependent factors shaping local
knowledge [14, 18, 27].
Passive recommendation proactively surfaces relevant informa-
tion without explicit user queries. Achieving effective passive rec-
ommendation in a local context necessitates understanding not
only individuals’ preferences but also the broader spatial cognition
that shapes their activity choices within a community. Although
recommender systems have been extensively studied in urban com-
puting [ 33], most existing approaches operate at a city-wide scale,
relying on conventional collaborative filtering or content-based
methods [ 3,19]. These methods, however, often fail to accurately
capture community-scale behavioral patterns or leverage residents’
unique cognitive understanding of places. As a result, they struggle
to provide localized recommendations that align with how people
perceive and navigate their immediate environments.
To address these challenges, we define the problem ofLocal
Life Information Accessibility(LLIA) as enabling residents to
efficiently obtain relevant, timely, and context-aware local infor-
mation through both active search and passive recommendation.
Distinct from traditional search or recommendation tasks, LLIA
requires jointly modeling spatial proximity, temporal relevance,
and cognitive familiarity at the neighborhood scale, which poses
unique technical and representational challenges. To support LLIA,
we proposeAskNearby, an AI-driven community system built on a
unified spatiotemporal retrieval and recommendation framework
powered by LLMs.AskNearbyconsists of two main components:
First, a three-layer Retrieval-Augmented-Generation (RAG) archi-
tecture that facilitates active information seeking by synergizing
GraphRAG(graph-based retrieval),VectorRAG(semantic simi-
larity retrieval), andGeoRAG(geographic-based retrieval); second,
a cognitive map model that captures users’ spatial preferences to
support passive, location-aware personalized recommendations.
Our main contributions are threefold:
•We define and formalize the LLIA problem, which encom-
passes both active search and passive recommendation tosupport spatiotemporal, context-aware access to neighborhood-
scale information.
•We design and developAskNearby, a novel community plat-
form that integrates a multi-layered LLM-based RAG archi-
tecture to enhance local information retrieval and a cognitive
map-based recommendation system.
•We validateAskNearbythrough extensive evaluations of
AskNearbyon real-world local community datasets, demon-
strating its superior performance in providing more accurate
and context-aware local information compared to baseline
methods.
2 Related Works
This section reviews prior research on spatiotemporal information
retrieval and recommendation, with a particular focus on meth-
ods that leverage textual data, natural language queries, and user-
generated content. We begin by reviewing traditional spatial and
temporal approaches, and subsequently, we discuss recent advances
driven by LLMs.
2.1 Local Information Retrieval and
Recommendation
Recent work on local information retrieval and recommendation
primarily addresses the spatial properties of Points of Interest (POIs)
and the temporal signals captured in user-generated or activity-
related content, such as check-ins, textual reviews, and event list-
ings. A range of techniques have been developed to improve re-
trieval accuracy and recommendation efficiency. Representative
approaches include context-aware modeling of user behavior [ 2,31],
spatial indexing and query optimization [ 15,16], time-sensitive rec-
ommendation strategies [ 3,20], and hybrid systems that combine
content-based and collaborative filtering [26].
However, existing approaches tend to focus on retrieving struc-
tured, global-scale content rather than localized, contextualized
knowledge. This leads to two key limitations. First, users often
need to interpret search results to extract meaningful local insights
manually. Second, most systems are not designed to handle open-
domain, natural-language queries about local life—queries that are
inherently ambiguous, context-rich, and highly personalized.
2.2 LLMs for Spatiotemporal Information
Retrieval and Recommendation
To address these challenges, recent work has turned to LLMs, which
have shown strong capabilities in understanding natural language
queries and reasoning over complex content [ 14]. Empirical studies
demonstrate that LLMs can effectively process time-series data,
geospatial inputs, and crowd-sourced textual content [ 7,10]. They
are also capable of inferring temporal relations from unstructured
text [ 28], improving geolocation representations through language
understanding [ 9], and identifying implicit spatial references em-
bedded within user expressions [ 25]. These emergent capabilities
position LLMs as highly promising tools for spatiotemporal infor-
mation access, particularly in local contexts where spatial, temporal,
and cognitive factors are deeply intertwined [14].
Capitalizing on these strengths, recent efforts have applied LLMs
to spatiotemporal information retrieval and recommendation tasks.

AskNearby: LLM-Based Neighborhood Info Retrieval & Recommendations GeoAI ’25, November 3–6, 2025, Minneapolis, MN, USA
For example, EvoRAG integrates retrieved user trajectories with
LLM reasoning to support personalized travel planning [?]. Tian
et al. [ 21] present a retrieval and re-ranking framework that uses
LLMs to identify similar environmental events from web and news
sources. Li et al. [ 13] propose an LLM-based framework for next
POI recommendation that preserves rich contextual signals from
location-based social network data, effectively addressing chal-
lenges such as cold-start and short user trajectories. LLMs have
also exhibited potential in interpreting nuanced spatial descrip-
tions (e.g., "nearby", "within walking distance") [ 8] and temporal
phrases (e.g., "open tonight", "happening this weekend") [ 29], which
traditional systems often struggle to handle.
Despite these notable advances, current LLM-based applications
typically lack a holistic spatiotemporal perspective [ 22,32]. Such
systems often prioritize textual semantics while overlooking the
complex interplay of time, space, user context, and cognitive spatial
familiarity. Consequently, significant challenges remain in deliver-
ing user-centric, context-aware, and personalized access to local
life information—especially within the dynamic, fine-grained envi-
ronments envisioned by the 15-minute city paradigm.
3 Methodology
3.1 Problem Definition: LLIA
LLIA refers to the ability of residents to efficiently obtain timely
and context-aware local information. It encompasses two comple-
mentary processes: active retrieval, which is based on explicit user
queries, and passive recommendation, which is driven by users’
spatial familiarity and local preference patterns.
Formally, a user 𝑢is characterized with spatial position ( 𝑠𝑢),
temporal context ( 𝑡𝑢), and frequently visited places ( 𝑝𝑢), which
approximate cognitive preferences. Given a spatiotemporal knowl-
edge base𝐾(containing POIs, events, and user-generated content
with geographic, temporal, and semantic metadata), LLIA aims to
return a set of relevant items 𝐼={𝑖 1,...,𝑖𝑛}that are both recent
and proximate. The LLIA function (𝐴) can be defined as:
𝐴(𝑢)=Retrieve(𝑄,𝑠 𝑢,𝐾)
|                {z                }
Active retrieval∪Recommend(𝑠 𝑢,𝑡𝑢,𝑝𝑢,𝐾)
|                           {z                           }
Passive recommendation.(1)
TheRetrieve(𝑄,𝑠 𝑢,𝐾)module retrieves candidate items relevant
to the user’s query 𝑄and current location 𝑠𝑢. The Recommend(𝑠 𝑢,𝑡𝑢,
𝑝𝑢,𝐾)module further ranks or selects results based on cognitive
relevance, incorporating semantic similarity, spatial proximity, and
public familiarity derived from𝑠 𝑢,𝑡𝑢, and𝑝𝑢.
3.2 System Overview
To enhance LLIA and provide local knowledge in a user-centric
manner, we proposeAskNearby, an AI-driven community system.
An overview of this system is illustrated in Figure 1.AskNearbycom-
bines a multi-layered RAG architecture with a cognitive map model
that captures users’ spatial preferences. This unified framework
enables both active information retrieval and passive, personalized
recommendations based on spatial familiarity. Lastly, it provides an
interactive user interface designed to facilitate real-time question-
answering, personalized recommendation posts, and map-based
visualization of nearby local information, as demonstrated in Fig-
ure 5 of Appendix A.3.3 RAGs for Local Information Access
To support open-domain, spatiotemporal local queries, we employ
a multi-layer RAG framework that decomposes and enriches the
user query via a structured processing pipeline. Upon receiving a
user-issued query 𝑄, an LLM is initially utilized to extract spatial
entities (e.g., "Futian Exhibition Center") and semantic intents (e.g.,
"restaurants", "entertainment") from the input.
Subsequently, a geospatial agent is invoked to query external
APIs (e.g., Gaode Maps) and construct geometric representations of
the identified locations in the form of points, polylines, or polygons.
These structured spatial objects, together with the user’s current lo-
cation, are passed to GeoRAG to retrieve geographically proximate
candidates.
In parallel, the semantic intent extracted from the query is used
to direct GraphRAG, which retrieves conceptually or relationally
relevant entities. Finally, the original query 𝑄is concatenated with
the graph-expanded semantic results and sent to VectorRAG to
retrieve semantically similar posts or documents in the vector space.
The overall retrieval pipeline is expressed as:
Retrieve(𝑄,𝑠 𝑢)=GeoRAG(Loc 𝑄,𝑠𝑢)
+GraphRAG(Sem 𝑄)
+VectorRAG(𝑄⊕Sem 𝐺),(2)
where Loc𝑄and Sem𝑄denote location and semantic intent ex-
tracted from 𝑄, and Sem𝐺is the graph-augmented semantic con-
text.
GeoRAGfocuses on accurate spatial computation and location-
based filtering. Backed by a spatial database (e.g.,PostGIS1), it sup-
ports geo-indexing, proximity querying, and spatial joins. GeoRAG
ensures that search results are geographically constrained and con-
textually relevant, such as identifying facilities within a specific
walking distance or filtering nearby services based on real-time
user location. Given a query location Loc𝑄(if available) or the
user’s current location 𝑠𝑢, and a candidate set of spatial entities 𝑆
that filtered from the spatiotemporal knowledge base 𝐾, GeoRAG
retrieves:
GeoRAG(Loc 𝑄,𝑠𝑢,𝑆)={𝑠∈𝑆|dist(Loc,𝑠)<𝜃},(3)
where Loc=Loc 𝑄if provided, otherwise Loc=𝑠𝑢. Here, dist(·)
denotes a spatial distance function (e.g., haversine), and 𝜃is a pre-
defined threshold (e.g., 1km).
GraphRAGleverages semantic graphs to retrieve conceptually
and relationally relevant information. Built on a graph database
(e.g.,NebulaGraph2), it supports efficient traversal, neighborhood
expansion, and relation-based filtering. The semantic graph encodes
relationships between entities such as POI categories, tags, events,
or user interests. Given a query intent Sem𝑄extracted from the
user input, GraphRAG identifies semantically related nodes and
paths based on graph connectivity and relation types. For a graph
𝐺—constructed based on the information from 𝐾—and query intent
Sem𝑄, the semantic expansion can be defined as:
GraphRAG(Sem 𝑄,𝐺)={𝑒∈𝐺|rel(Sem 𝑄,𝑒)∈𝑅},(4)
1https://postgis.net/
2https://www.nebula-graph.com.cn/

GeoAI ’25, November 3–6, 2025, Minneapolis, MN, USA Niu et al.
Figure 1: System overview ofAskNearby, combining LLM-based object extraction, multi-source retrieval (geospatial, vector,
graph), and a cognitive map model to support active information retrieval and passive local recommendations.
where𝑅denotes a set of predefined semantic or relational edges
in the graph, and rel(·) indicates the presence of a meaningful
semantic connection.
VectorRAGenables semantic-level retrieval by supporting ap-
proximate nearest neighbor search over candidate contents. Vector-
RAG refines results from the outputs of GeoRAG and GraphRAG
by ranking them based on semantic similarity to the enhanced
query. Specifically, we concatenate the original query 𝑄with graph-
augmented semantic context Sem𝐺to form an enriched input 𝑄′=
𝑄⊕Sem𝐺.
We utilizepgvector3as the vector similarity search engine, al-
lowing efficient storage and querying of dense embeddings within
a PostgreSQL-compatible environment. Both user queries and local
content are embedded into high-dimensional vector representa-
tions through LLM-based encoders. Specifically, we adopt thebge-
large-zh-v1.5model released by the Beijing Academy of Artificial
Intelligence [ 23], which has demonstrated strong performance in
semantic understanding tasks for Chinese text. We use Euclidean
distance as the similarity metric to retrieve the most semantically
relevant results from the filtered candidate set 𝑉′composed of these
outputs from GeoRAG and GraphRAG. Formally, with encoder 𝜙(·)
3https://github.com/pgvector/pgvectorand candidate set𝑉′derived from previous modules:
VectorRAG(𝑄,𝑉′)={𝑣∈𝑉′|sim(𝜙(𝑄⊕Sem 𝐺),𝜙(𝑣))>𝛿},
(5)
where the function sim(·) denotes a similarity measure derived as
the Euclidean distance between embeddings, and the threshold 𝛿
controls the minimum similarity required for a candidate 𝑣to be
selected.
Beyond active retrieval, personalized and context-aware recom-
mendations are also essential for enhancing accessibility to local
life information. To this end, we introduce a cognitive map model
that captures individual user preferences across space and time, as
described in the following subsection.
3.4 Cognitive Map Model for Personalized
Recommendation
To support contextualized and personalized recommendations in
local life scenarios, we propose a cognitive map model that captures
how users internally perceive and associate meaning with urban
spaces over time. Users’ preferences vary not only by spatial proxim-
ity but also by how they functionally and socially interpret different
places. To translate these human-centered cognitive preferences
into actionable recommendations, we define the recommendation

AskNearby: LLM-Based Neighborhood Info Retrieval & Recommendations GeoAI ’25, November 3–6, 2025, Minneapolis, MN, USA
score for an information item𝑖as:
Ψ(𝑠𝑢,𝑡𝑢,𝑝𝑢,𝑖)=𝑓 sem(𝑠𝑢,𝑡𝑢,𝑝𝑢,𝑖)𝛼·𝑓dist(𝑑(𝑠𝑢,𝑖))𝛽·𝑓pop(𝑖)𝛾,(6)
where𝛼,𝛽,𝛾 control the relative contributions of semantic rele-
vance, spatial proximity, and public familiarity.
In this formulation, the overall recommendation score is com-
puted as the product of three components: thesemantic relevance
term𝑓sem(𝑠𝑢,𝑡𝑢,𝑝𝑢,𝑖), which measures the similarity between the
user’s cognitive profile and the temporal functional semantics of the
item’s location; thespatial proximityterm 𝑓dist(𝑑(𝑠𝑢,𝑖)), which
models geographic relevance via a distance-decay function based on
the spatial distance between the user and the item; and thepublic
familiarityterm 𝑓pop(𝑖), which reflects the aggregated popular-
ity or visitation frequency of the location, capturing its collective
cognitive salience.
To reflect the relative importance of different components in the
recommendation model, we assign each factor an exponent 𝛼,𝛽,
and𝛾, which serve as soft weights in the multiplicative scoring
function. We initialize these three weights equally at 1.0 to avoid
imposing any prior preference, allowing semantic relevance, spatial
proximity, and public familiarity to contribute equally to the final
ranking. We acknowledge that a more optimal weight combination
could potentially be found through grid search on the validation
set, and we leave this for future work.
Figure 2 illustrates the conceptual model behind this scoring
framework, showing how the semantic, spatial, and popularity
signals interact to determine which content to recommend to the
user in different contexts.
LongitudeLatitudeTime Window
Semanctic Relevance
Spatial ProximityPublic FamiliarityTF-IWF
Figure 2: Illustration of the recommendation framework in
AskNearby, combining semantic relevance, spatial proximity,
and public familiarity to compute personalized scores.
Semantic relevance.To model how users cognitively perceive
and associate meaning with places, we implement the semantic
component using a TF-IWF (Term Frequency-Inverse Word Fre-
quency) based representation [ 6], which captures the time-varying
functional semantics of each location. This scheme identifies place
attributes that are locally prominent yet globally distinctive across
different time periods. Formally, the TF-IWF score is computed as:
TF-IWF𝑘,𝑔=𝐴𝑘,𝑔Í
𝑖𝐴𝑖,𝑔·logÍ
ℎ𝐴𝑘,ℎÍ
𝑖Í
ℎ𝐴𝑖,ℎ
,(7)where𝐴𝑘,𝑔denotes the count of functional attribute 𝑘(e.g.,
restaurant, office, park) at location 𝑔during a specific time window.
For example, a location may exhibit office-related semantics during
the day and shift toward dining in the evening.
The semantic similarity 𝑓sem(𝑠𝑢,𝑡𝑢,𝑝𝑢,𝑖)is computed as the inner
product between the TF-IWF vector of the item’s location at time
𝑡𝑢and the user’s cognitive profile. This profile is constructed by
aggregating the TF-IWF vectors of the locations in 𝑝𝑢, the set of
frequently visited places by user 𝑢. Each location’s contribution is
weighted based on its temporal proximity to the current time 𝑡𝑢. In
addition, the TF-IWF vector of the user’s currently location 𝑠𝑢at
time𝑡𝑢is included to capture short-term contextual relevance.
This design allows the semantic matching to reflect both the
user’s long-term spatial familiarity and the immediate context of
their current activity.
Spatial proximity.We model spatial relevance using a distance-
decay function 𝑓dist(𝑑)=exp(−𝜆 𝑑·𝑑(𝑠𝑢,𝑖)), where𝑑(𝑠𝑢,𝑖)is the
geographical distance between user and item, and 𝜆𝑑controls the de-
cay rate. This formulation encourages recommending closer items
while still allowing semantically relevant distant items to appear.
Public familiarity.To account for socially shared knowledge,
𝑓pop(𝑖)reflects how frequently item 𝑖’s location has been visited
by the general public. This component promotes widely recog-
nized places and enhances collective cognitive relevance, thereby
increasing the exposure of popular and trustworthy content in
recommendations.
By integrating personalized semantics, spatial constraints, and
collective familiarity, our model provides passive recommendations
that are both user-centric and socially grounded. This allows the
system to suggest nearby and timely content that reflects both
individual interests and shared patterns in the local community.
4 Experiments and Results
4.1 Data Description
To facilitate a robust empirical evaluation ofAskNearbyand its
comparison with other LLM-based applications, we constructed a
dataset of real-world, location-based user-generated content. This
data was sourced from RedNote, one of China’s foremost social
media platforms for local discovery and lifestyle sharing. The study
area is set in Shenzhen, a large metropolitan city in China. We
programmatically collected geotagged posts associated with ma-
jor landmarks (e.g., Nantou Ancient Town, Wutong Mountain) by
targeting location-specific keywords.
To mitigate potential biases from RedNote’s native recommen-
dation algorithm, we implemented a three-stage data collection
and processing methodology. The process began with a broad key-
word search across the study area to gather a comprehensive and
unfiltered corpus of candidate posts. This initial corpus was then
subjected to a rigorous cleaning phase, which included deduplica-
tion, the removal of advertisements and spam, and sample-based
verification of geographic information accuracy. Finally, from this
refined corpus, we performed random sampling to constitute our
final dataset, thereby ensuring it is a representative, rather than a
platform-biased, sample of user-generated content.
The resulting dataset comprises approximately 20,000 posts and
encompasses a diverse range of popular activity areas, including

GeoAI ’25, November 3–6, 2025, Minneapolis, MN, USA Niu et al.
historical sites like Nantou Ancient Town and natural attractions
such as Wutong Mountain. This selection ensures comprehensive
representation across a spectrum of urban functions, from com-
mercial and recreational zones to key transportation hubs. Each
record contains essential metadata such as post content, timestamp,
user attributes (e.g., type), interaction metrics, semantic tags, and
geographic coordinates. Table 1 provides a summary of the data
schema.
Table 1: Key fields in the RedNote dataset
Field Description
Title, Content Textual description of the post
Timestamp Posting date and time
Author, Type Username and whether verified or not
Likes, Comments Interaction counts
Semantic Tags Keywords, topics, sentiment labels
Location Name Place name extracted or tagged
Longitude, Latitude Geographic coordinates (WGS84)
Media Type Post format (text, image, video)
Original or Repost Indicates whether the post is original content
This dataset serves as a rich source of localized information
for benchmarking howAskNearbyretrieves and recommends spa-
tiotemporally relevant content compared to baseline LLM systems
that lack grounded urban context. Furthermore, it enables fine-
grained evaluation of user intent understanding, place-aware rank-
ing, and recommendation diversity. The spatial distribution of the
data is visualized in Figure 3, where the heatmap highlights the
concentration of activity around major commercial and cultural
hubs, illustrating the high spatial resolution and urban relevance
of the dataset.
Size & color∝data volume
Figure 3: Spatial distribution of RedNote posts in Shenzhen.
4.2 Experimental Setup
Based on the dataset, we designed a set of local-life-oriented queries
to assess the system’s retrieval and generation performance.Notably,
both the system and the dataset were originally in Chinese. For this
paper, we provide a translated version in English to illustrate ourapproach. To ensure the linguistic alignment, we adopt ChatGLM-
4 [30], one of the most advanced Chinese large language models
available.
4.3 Evaluation Metrics
We use metrics covering two aspects to evaluate the retrieval and
recommendation, respectively.
For retrieval ability, we usePrecision,Normalized Discounted
Cumulative Gain (NDCG),Hallucination Rate (HR), andSpatial-
Temporal Relevance (STR)to evaluate.
•Precision: measures the relevance of retrieved posts. High
precision indicates fewer irrelevant results. Since users usu-
ally only focus on the first few results, we adopt Precision@4
in this study.
•Normalized Discounted Cumulative Gain (NDCG): eval-
uates ranking quality by incorporating graded relevance
scores for each post (2 = highly relevant, 1 = moderately
relevant, 0 = irrelevant). The metric computes a weighted
cumulative gain for the top K results (here we use k=4) and
normalizes it against the ideal ranking.
Specifically,
NDCG@4=DCG@4
IDCG@4(8)
DCG@4=4∑︁
𝑖=12rel𝑖−1
log2(𝑖+1)(9)
IDCG@4=4∑︁
𝑖=12rel★
𝑖−1
log2(𝑖+1)(10)
Here, rel★
𝑖represents the relevance scores in the ideal de-
scending order.
•Hallucination Rate (HR): proportion of responses gener-
ated by the model with non-existent or incorrectly attrib-
uted local information (e.g., fictitious attractions or incorrect
prices).
•Spatial-Temporal Relevance (STR): quantifies whether
retrieved results are geographically constrained within the
queried "proximity range" (e.g., a 1km living radius) and
satisfy implicit temporal constraints (e.g., operational hours).
Each result is scored 1 if both conditions are met; otherwise,
it is 0.
•Answer Quality (AQ): the overall integrity, coherence, and
fluency of generated answers.
•Match Score (MS): the degree of alignment between the
answer and the query’s intent, reflecting how effectively the
response resolves the user’s underlying need.
Furthermore, we introduce two LLM-evaluated metrics to com-
plement several aspects that are hard to quantify: : (1)Answer
Quality (AQ): the overall integrity, coherence, and fluency of gen-
erated answers. (2)Match Score (MS): the degree of alignment be-
tween the answer and the query’s intent, reflecting how effectively
the response resolves the user’s underlying need. The prompts of
LLMs for measuringAQandMSare detailed in Appendix B.
For recommendation ability, we utilizedHit Rate (Hit@K),
Normalized Discounted Cumulative Gain (NDCG@K)and
Mean Reciprocal Rank (MRR)to measure.

AskNearby: LLM-Based Neighborhood Info Retrieval & Recommendations GeoAI ’25, November 3–6, 2025, Minneapolis, MN, USA
•Hit Rate (Hit@K): measures whether any ground-truth
item appears within the top 𝐾recommended items. It reflects
the coverage of relevant results, regardless of position. We
report Hit@5 and Hit@10 in the experiment.
•Normalized Discounted Cumulative Gain (NDCG@K):
evaluates ranking quality by rewarding relevant items ap-
pearing earlier in the list. It is computed as described in the
retrieval metrics, and we report NDCG@5 and NDCG@10.
•Mean Reciprocal Rank (MRR): the reciprocal of the rank
at which the first relevant item appears in the recommen-
dation list. A higher MRR indicates that relevant content is
shown earlier, enhancing user satisfaction.
These metrics effectively reflect user satisfaction in life-circle
scenarios by measuring whether relevant posts appear early within
a limited number of local recommendations.
Considering the nuanced judgment required by the evaluation,
we employed a human evaluation methodology for the rigorous
and objective comparison of our model against the baselines. The
process involved three trained evaluators who assessed the outputs
from all models, presented in a randomized and anonymized order
to prevent bias. For relevance-based metrics like Precision@4 and
NDCG@4, evaluators assigned relevance scores to each retrieved
item, which were then used to compute the final values. For quali-
tative metrics such as Hallucination Rate (HR) and Answer Quality
(AQ), they assigned scores directly based on predefined criteria
including factual accuracy, spatiotemporal adherence, and contex-
tual alignment. To ensure reliability, the final score for each metric
was the average from the three evaluators, with inter-annotator
agreement calculated and any significant discrepancies resolved
through discussion.
4.4 Ablation Study
We conducted an ablation study to systematically evaluate the
retrieval and recommendation capabilities of our system, respec-
tively. This approach was chosen because our system is specifically
engineered to address the novel LLIA problem by integrating dy-
namic user context, cognitive maps, and spatial-semantic features.
Standard recommenders intrinsically lack these capabilities, and
adapting them for this task would require non-trivial modifications,
creating a significant risk of comparison bias. Therefore, an abla-
tion study provides a more direct and insightful evaluation of each
component in both retrieval and recommendation modules.
Results are summarized in Table 2 and Table 3. For retrieval,
the full three-layer RAG (GraphRAG + VectorRAG + GeoRAG)
achieves the best performance. Removing GraphRAG moderately
reduces Precision@4 and NDCG@4, while excluding GeoRAG leads
to the largest performance drop, with STR falling from 83.8% to
58.7% and hallucination rate rising, underscoring its role in ground-
ing responses. VectorRAG alone performs worst across all metrics,
showing that semantic similarity retrieval is insufficient without
geographic and relational constraints. Through the experiments,
we have validated the necessity of combining all three retrieval
strategies to ensure both contextual relevance and geographic cor-
rectness in real-world use.
For recommendation, combining spatial proximity, public famil-
iarity, and semantic relevance yields the best results. Droppingsemantic or public cues degrades ranking quality and early-hit ac-
curacy, while relying on spatial proximity alone performs worst,
confirming that spatial context must be complemented with social
and semantic signals.
4.5 Comparison Experiment
We then focus on evaluating retrieval performance. Our system is
compared against several representative baseline models, includ-
ing both LLM-based architectures. In addition to state-of-the-art
general-purpose LLMs such as GPT-4o4, we also include Chinese-
developed models like DeepSeek5and Qwen6. Since these models
do not inherently possess localized knowledge, we construct an
external knowledge base for each and enable retrieval to ensure a
fair comparison.
To reflect real-world usability, we further compareAskNearby
with domain-specific or online map platforms, including RedNote7,
as well as the most widely adopted local map applications in China,
such as Gaode Maps8and Baidu Maps9. These platforms have re-
cently integrated advanced LLMs (e.g., DeepSeek-R1) to support
natural language-based local information search and summariza-
tion. We conduct manual searches through the platform interface
and collect the corresponding returned results.
Here are the introductions of our baseline models:
•GPT-4o: State-of-the-art multilingual LLMs with strong gen-
eral reasoning and dialogue capabilities.
•DeepSeek R1andQwen3 Turbo: High-performance Chi-
nese LLMs optimized for local language understanding. These
models demonstrate strong reasoning capabilities and com-
plex Chinese instruction-following behavior, with signifi-
cantly improved human preference alignment compared to
previous generations.
•RedNote: A local lifestyle and information-sharing platform
with user-generated content and keyword-based search.
•Gaode Maps, Baidu Maps: Leading map applications in
China that offer location-based search with LLM-enhanced
natural language interfaces.
Table 4 shows the evaluation results of our system.AskNearby
outperforms most baselines across retrieval metrics. However, in
precision, it ranks second (75.6%) slightly below RedNote (78.3%).
This is likely attributed to the dominance of simple keyword match-
ing tasks in the experimental queries (e.g., "Starbucks"), where Red-
Note’s keyword-based engine excels. In contrast, RedNote struggles
with queries requiring complex intent or spatiotemporal constraints
(e.g., "a quiet coffee shop for afternoon work") whileAskNearbyex-
cels with its superior semantic understanding capabilities, evident
from its much higher NDCG@4 score (0.96 vs. 0.88). Notably,AskN-
earbyfar surpasses all baselines in STR, underscoring its ability to
semantically and spatiotemporally retrieve relevant results—critical
in real-world local information access. In addition,AskNearbyex-
cels in generating high-quality responses with minimal hallucina-
tions. It effectively retrieves and summarizes relevant knowledge
4https://chatgpt.com/
5https://www.deepseek.com/
6https://chat.qwen.ai/
7https://www.xiaohongshu.com/
8https://gaode.com/
9https://map.baidu.com/

GeoAI ’25, November 3–6, 2025, Minneapolis, MN, USA Niu et al.
Table 2: Ablation study on different RAG-module combinations inAskNearby. The first row is the full model. Subsequent rows
show the effect of removing one or more modules.
Method Precision@4 (%) NDCG@4 HR (%) STR (%) AQ MS
AskNearby (All RAGs) 75.6 0.96 2.5 83.8 3.9 4.2
Geo + Vector (— GraphRAG)70.8 0.91 3.2 82.9 3.6 3.8
Graph + Vector (— GeoRAG)66.1 0.83 4.5 58.7 3.4 3.2
VectorRAG only (— GeoRAG, — GraphRAG)59.7 0.78 5.1 53.6 3.1 2.9
Table 3: Ablation results of theAskNearbyrecommendation module (S = spatial proximity, P = public familiarity, Sem = semantic
relevance).
Method Hit@5 Hit@10 NDCG@5 NDCG@10 MRR
S + P + Sem 0.612 0.708 0.482 0.534 0.421
S + Sem 0.587 0.682 0.455 0.506 0.386
S + P 0.573 0.669 0.443 0.493 0.397
S only 0.542 0.638 0.412 0.461 0.357
Table 4: Evaluation results ofAskNearbyvs. baseline models
(Bold: best, underline: second-best )
Method Precision@4↑(%) NDCG@4↑HR↓(%) STR↑
GPT-4o 58.3 0.86 15.8 41.7
DeepSeek-R1 72.8 0.92 15.8 64.2
Qwen-3-turbo 70.8 0.92 15.0 62.5
RedNote78.30.88 13.1 56.9
Gaode maps 50.0 0.65 9.3 41.0
Baidu maps 65.0 0.71 3.0 50.0
AskNearby (Ours)75.6 0.96 2.5 83.8
grounded in local context, leading to informative and trustworthy
answers.
4.6 Qualitative Analysis
To demonstrate the practical effectiveness of our system, we further
conduct a qualitative analysis in real-world local information sce-
narios. Figure 4 presents a representative case in which a user poses
the context-specific query: "Where can I find healthcare services
inside the neighborhood?". We compare the response generated by
AskNearbyagainst a baseline LLM-based application.
As the figure illustrates, the baseline application, while identi-
fying semantically related content, ultimately hallucinates a non-
existent address. This failure stems from a lack of accurate, localized
knowledge grounding, which can mislead the user. In stark contrast,
AskNearbyeffectively synthesizes information from relevant posts
to extract and present the correct service location, substantiated
with supporting details such as contact information.
In addition to this example, generic LLMs, even when enhanced
with retrieval capabilities, often suffer from hallucination when
dealing with fine-grained, up-to-date local information. This further
indicates the superiority of our AskNearby in satisfying the needs
for the retrieval and recommendation of local information within
the life circle.
How do I get porteringservices in Nantou Ancient Town? Where is it?Hello! If you’re looking for porteringservices in the Nantou Ancient Town, you can contact the “FortunePorter Team”. This is a team of local movers and haulers who specialize in meeting the moving needs of the Ancient Town. Their base turns out to be there at Nantou Middle School! If you want porteringservice, just go there to find it~!
Nantou Middle School
FortunePorter TeamNantou Middle SchoolThe base of “Fortune Porter Team” is Here~ Come to us if you need help! NeighborhoodInformation
Nantou AncientTown community has a standardized porter team to provide convenient services:No. 54 XingmingStreet, Nantou Ancient Town (XinweiSuMoving Service Department, Nanshan District, Shenzhen)False address caused by hallucinations
Correct and high-quality answer
Problems of Real Local Life
Figure 4: Answers to a local life problem from AskNearby
and Baidu maps (powered by DeepSeek-R1).
4.7 Deployed System Performance
To validate the effectiveness and scalability ofAskNearby, we con-
ducted extensive field deployments in two diverse real-world envi-
ronments: a university campus (PKU-SZ) and a high-density, mixed-
use urban community (Nantou Ancient City). Our field deployments
yielded significant positive outcomes in two diverse environments.
On a university campus,AskNearbysaw rapid adoption, achiev-
ing a 46% uptake among new students within 30 days. In a dense
urban community (Nantou, 38,000 residents/km2), the system fos-
tered significant hyperlocal engagement, as evidenced by residents
initiated 4.1 queries per month on average, enabling 1,472 offline
interactions.
Also, we conduct manual evaluation by inviting 40 regular users
(User) and 10 domain experts (Expert) in urban planning to as-
sessAskNearbyagainst baselines based on their individual queries.

AskNearby: LLM-Based Neighborhood Info Retrieval & Recommendations GeoAI ’25, November 3–6, 2025, Minneapolis, MN, USA
Table 5: Deployed system performance ofAskNearbyvs. base-
line models (Bold: best, underline: second-best )
Method AQ↑MS↑User↑Expert↑
GPT-4o 3.5 3.8 3.9 3.1
DeepSeek-R1 3.7 4.0 4.0 4.3
Qwen-3-turbo 3.6 3.7 3.2 3.3
RedNote 2.2 3.1 2.7 2.5
Gaode maps 3.1 3.2 2.8 3.2
Baidu maps 3.8 3.9 3.7 3.8
AskNearby (Ours) 3.9 4.2 4.6 4.5
Performance was measured across four key dimensions: AQ and
MS, alongside manual ratings from Users and Experts. The results,
summarized in Table 5, demonstrate the superiority of our system,
confirming its effectiveness in real-world scenarios.
This robust performance, validated in both large-scale deploy-
ments and controlled evaluations, underscores the real-world value
of our approach, enhancing information accessibility and serving
as a social connector that fosters neighborhood interaction.
5 Conclusion
In this paper, we introduce the problem of LLIA, which focuses
on enabling residents to efficiently acquire relevant, timely, and
context-aware information within their neighborhood. Unlike tra-
ditional search or recommendation systems that overlook fine-
grained spatiotemporal dynamics and user-specific spatial cogni-
tion, LLIA highlights the importance of integrating geographic
proximity, temporal constraints, and personalized perceptions in
local information access. To address this challenge, we propose
AskNearby, an AI-driven community platform integrating a three-
layer retrieval-augmented generation framework with a cognitive
map-based recommendation model.
Experiments on real-world datasets demonstrate thatAskNearby
significantly outperforms both LLM-based baselines and existing
local service platforms in retrieval accuracy, contextual alignment,
and hallucination mitigation. Qualitative studies and real-world
deployments further validate its effectiveness, demonstrating both
accurate localized knowledge grounding and the empowerment of
residents with hyperlocal knowledge. By supporting both active
search and passive recommendation at the neighborhood scale, as
well as the outstanding ability of understanding natural language,
AskNearbytransforms static POIs into more local scenarios, allow-
ing residents to know not only "what is there," but more importantly,
"what it is like there," thereby effectively bridges the gap between
physical proximity and true psychological and lifestyle accessibility.
This brings the 15-minute city vision closer to reality. Moreover,
unearthing fine-grained urban information overlooked by tradi-
tional maps, our work activates rich local resources, transforming
communities from static geographical entities into dynamic socio-
economic networks, greatly enriching the concept of the "15-minute
city" and opening new possibilities for human-centric spatial com-
puting in urban environments.
While ourAskNearbysystem shows promising results, we ac-
knowledge certain limitations and identify key directions for future
work. We plan to advance our research along three main fronts.First, we aim to refine the core cognitive map model, focusing on
more sophisticated, adaptive mechanisms to capture individual
user nuances and their evolving preferences. Second, to enhance
the scientific rigor and generalizability of our findings, we will
significantly expand our data foundation and refine our evalua-
tion methodologies. This involves diversifying data sources and
geographical coverage beyond initial settings, alongside exploring
more robust assessment protocols. Finally, we will focus on enhanc-
ing the practical utility and real-world applicability ofAskNearby.
This includes enriching its knowledge base with a wider array of
information sources to improve overall accuracy and versatility,
and conducting broader deployments to validate its effectiveness
across diverse urban contexts and contribute to the development
of intelligent localized services.
Acknowledgments
This research was supported by Dr. Qi Shu and theShu Qi Youth
Innovation Leadership Charitable Trust. The authors would like to
express sincere gratitude for their generous support and valuable
guidance.
References
[1]Timur Abbiasov, Cate Heine, Sadegh Sabouri, Arianna Salazar-Miranda, Paolo
Santi, Edward Glaeser, and Carlo Ratti. 2024. The 15-minute city quantified using
human mobility data.Nature Human Behaviour8, 3 (2024), 445–455.
[2]Mohammad Aliannejadi and Fabio Crestani. 2018. Personalized context-aware
point of interest recommendation.ACM Transactions on Information Systems
(TOIS)36, 4 (2018), 1–28.
[3]Mohammad Aliannejadi, Dimitrios Rafailidis, and Fabio Crestani. 2019. A joint
two-phase time-sensitive regularized collaborative ranking model for point of
interest recommendation.IEEE Transactions on Knowledge and Data Engineering
32, 6 (2019), 1050–1063.
[4]Matteo Bruno, Hygor Piaget Monteiro Melo, Bruno Campanelli, and Vittorio
Loreto. 2024. A universal framework for inclusive 15-minute cities.Nature Cities
1, 10 (2024), 633–641.
[5]Shahaf Donio and Eran Toch. 2025. Neighborhood Disparities in Smart City
Service Adoption.arXiv preprint arXiv:2501.04363(2025).
[6]Zhipeng Gui, Yunzeng Sun, Le Yang, Dehua Peng, Fa Li, Huayi Wu, Chi Guo,
Wenfei Guo, and Jianya Gong. 2021. LSI-LSTM: An attention-aware LSTM for
real-time driving destination prediction by considering location semantics and
location importance of trajectory points.Neurocomputing440 (2021), 72–88.
[7]Qiuhan Han, Atsushi Yoshikawa, and Masayuki Yamamura. 2025. Adapting Large
Language Model for Spatio-Temporal Understanding in Next Point-of-Interest
Prediction. InICASSP 2025-2025 IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP). IEEE, 1–5.
[8]Erum Haris, Anthony G Cohn, and John G Stell. 2024. Exploring spatial repre-
sentations in the historical lake district texts with llm-based relation extraction.
arXiv preprint arXiv:2406.14336(2024).
[9]Junlin He, Tong Nie, and Wei Ma. 2025. Geolocation representation from large lan-
guage models are generic enhancers for spatio-temporal learning. InProceedings
of the AAAI Conference on Artificial Intelligence, Vol. 39. 17094–17104.
[10] Zongcai Huang, Peng Peng, Feng Lu, and He Zhang. 2025. An LLM-Based
Method for Quality Information Extraction From Web Text for Crowed-Sensing
Spatiotemporal Data.Transactions in GIS29, 1 (2025).
[11] Amir Reza Khavarian-Garmsir, Ayyoob Sharifi, and Ali Sadeghi. 2023. The
15-minute city: Urban planning and design efforts toward creating sustainable
neighborhoods.Cities132 (2023), 104101.
[12] Boyang Li and Wenjia Zhang. 2024. Enhancing Pedestrian Route Choice Models
Through Maximum-Entropy Deep Inverse Reinforcement Learning With Indi-
vidual Covariates (MEDIRL-IC).IEEE Transactions on Intelligent Transportation
Systems25, 12 (2024), 20446–20463. doi:10.1109/TITS.2024.3457680
[13] Peibo Li, Maarten de Rijke, Hao Xue, Shuang Ao, Yang Song, and Flora D Salim.
2024. Large language models for next point-of-interest recommendation. In
Proceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval. 1463–1472.
[14] Zhonghang Li, Lianghao Xia, Jiabin Tang, Yong Xu, Lei Shi, Long Xia, Dawei
Yin, and Chao Huang. 2024. Urbangpt: Spatio-temporal large language models.
InProceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining. 5351–5362.

GeoAI ’25, November 3–6, 2025, Minneapolis, MN, USA Niu et al.
[15] Hui Luo, Jingbo Zhou, Zhifeng Bao, Shuangli Li, J. Shane Culpepper, Haochao
Ying, Hao Liu, and Hui Xiong. 2020. Spatial Object Recommendation with Hints:
When Spatial Granularity Matters. InProceedings of the 43rd International ACM
SIGIR Conference on Research and Development in Information Retrieval(Virtual
Event, China)(SIGIR ’20). Association for Computing Machinery, New York, NY,
USA, 781–790. doi:10.1145/3397271.3401090
[16] Shunmei Meng, Huihui Wang, Qianmu Li, Yun Luo, Wanchun Dou, and Shaohua
Wan. 2018. Spatial-Temporal Aware Intelligent Service Recommendation Method
Based on Distributed Tensor factorization for Big Data Applications.IEEE Access
6 (2018), 59462–59474. doi:10.1109/ACCESS.2018.2872351
[17] Carlos Moreno, Zaheer Allam, Didier Chabaud, Catherine Gall, and Florent
Pratlong. 2021. Introducing the “15-Minute City”: Sustainability, resilience and
place identity in future post-pandemic cities.Smart cities4, 1 (2021), 93–111.
[18] Hang Ni, Fan Liu, Xinyu Ma, Lixin Su, Shuaiqiang Wang, Dawei Yin, Hui Xiong,
and Hao Liu. 2025. TP-RAG: Benchmarking Retrieval-Augmented Large Lan-
guage Model Agents for Spatiotemporal-Aware Travel Planning.arXiv preprint
arXiv:2504.08694(2025).
[19] Lara Quijano-Sánchez, Iván Cantador, María E Cortés-Cediel, and Olga Gil. 2020.
Recommender systems for smart cities.Information systems92 (2020), 101545.
[20] Ting Shen, Haiquan Chen, and Wei-Shinn Ku. 2018. Time-aware location se-
quence recommendation for cold-start mobile users. InProceedings of the 26th
ACM SIGSPATIAL international conference on advances in geographic information
systems. 484–487.
[21] Yuanyuan Tian, Wenwen Li, Lei Hu, Xiao Chen, Michael Brook, Michael Brubaker,
Fan Zhang, and Anna K. Liljedahl. [n. d.]. Advancing Large Language Models
for Spatiotemporal and Semantic Association Mining of Similar Environmental
Events. ([n. d.]). doi:10.1111/tgis.13282
[22] Tianxing Wang and Can Wang. 2024. Embracing LLMs for Point-of-Interest
Recommendations.IEEE Intelligent Systems39, 1 (2024), 56–59. doi:10.1109/MIS.
2023.3343489
[23] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023.
C-Pack: Packaged Resources To Advance General Chinese Embedding.
arXiv:2309.07597 [cs.CL]
[24] Fengli Xu, Qi Wang, Esteban Moro, Lin Chen, Arianna Salazar Miranda, Marta C
González, Michele Tizzoni, Chaoming Song, Carlo Ratti, Luis Bettencourt, et al .
2025. Using human mobility data to quantify experienced urban inequalities.
Nature Human Behaviour(2025), 1–11.
[25] Wenping Yin, Yong Xue, Ziqi Liu, Hao Li, and Martin Werner. 2025. LLM-
enhanced disaster geolocalization using implicit geoinformation from multimodal
data: A case study of Hurricane Harvey.International Journal of Applied Earth
Observation and Geoinformation137 (2025), 104423.
[26] Phatpicha Yochum, Liang Chang, Tianlong Gu, and Manli Zhu. 2020. Linked
Open Data in Location-Based Recommendation System on Tourism Domain: A
Survey.IEEE Access8 (2020), 16409–16439. doi:10.1109/ACCESS.2020.2967120
[27] Dazhou Yu, Riyang Bao, Gengchen Mai, and Liang Zhao. 2025. Spatial-RAG: Spa-
tial Retrieval Augmented Generation for Real-World Spatial Reasoning Questions.
arXiv preprint arXiv:2502.18470(2025).
[28] Xinli Yu, Zheng Chen, Yuan Ling, Shujing Dong, Zongyi Liu, and Yanbin Lu. 2023.
Temporal data meets LLM–explainable financial time series forecasting.arXiv
preprint arXiv:2306.11025(2023).
[29] Chenhan Yuan, Qianqian Xie, Jimin Huang, and Sophia Ananiadou. 2024. Back to
the future: Towards explainable temporal reasoning with large language models.
InProceedings of the ACM Web Conference 2024. 1963–1974.
[30] Team GLM: Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan
Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning
Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Jingyu
Sun, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang,
Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao,
Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang,
Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai
Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao
Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and
Zihan Wang. 2024. ChatGLM: A Family of Large Language Models from GLM-
130B to GLM-4 All Tools. arXiv:2406.12793 [cs.CL] https://arxiv.org/abs/2406.
12793
[31] Pengpeng Zhao, Anjing Luo, Yanchi Liu, Jiajie Xu, Zhixu Li, Fuzhen Zhuang,
Victor S. Sheng, and Xiaofang Zhou. 2022. Where to Go Next: A Spatio-Temporal
Gated Network for Next POI Recommendation.IEEE Transactions on Knowledge
and Data Engineering34, 5 (2022), 2512–2524. doi:10.1109/TKDE.2020.3007194
[32] Zihuai Zhao, Wenqi Fan, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen
Wen, Fei Wang, Xiangyu Zhao, Jiliang Tang, and Qing Li. 2024. Recommender
Systems in the Era of Large Language Models (LLMs).IEEE Transactions on
Knowledge and Data Engineering36, 11 (2024), 6889–6907. doi:10.1109/TKDE.
2024.3392335
[33] Yu Zheng, Licia Capra, Ouri Wolfson, and Hai Yang. 2014. Urban computing:
concepts, methodologies, and applications.ACM Transactions on Intelligent
Systems and Technology (TIST)5, 3 (2014), 1–55.[34] Yu Zheng and Xing Xie. 2011. Learning travel recommendations from user-
generated GPS traces.ACM Transactions on Intelligent Systems and Technology
(TIST)2, 1 (2011), 1–29.
A Demonstration of Our AskNearby System
Figure 5 presents the screenshots of our developed AskNearby
system, which is designed to enhance local life information accessi-
bility through both recommendation and retrieval functionalities.
Subfigure (a) illustrates theRecommendation Page, where
users receive personalized local content suggestions based on lo-
cation, time, and cognitive preferences. The left panel features a
vertically scrollable list of recommended posts concerning nearby
locations or events, detailed with text descriptions, timestamps,
spatial distances, and multimedia content. Adjacent to this list, an
interactive map interface is provided in the right panel. On this
map, users can adjust the location pointer to refresh their current
view and receive recommendations for different areas.
Subfigure (b) shows theRetrieval Page, which supports natural-
language-based active querying. Users can input open-domain
questions about their surroundings, such as "Where are the toilets
nearby?", and the system retrieves relevant spatial entities from the
spatiotemporal knowledge base. The retrieval results are displayed
in textual format, providing detailed descriptions, and visually on
the map, allowing users to explore spatial relations interactively.
The system also suggests example queries to facilitate information
discovery.
B LLM Evaluation Prompts
B.1 LLM Evaluation Prompt for Local
Information Retrieval
To assess AQ and MS metrics, we utilized Gemini 2.5 as the evalua-
tor, leveraging its established high correlation with human judg-
ment. The selection of an external model, distinct from our internal
ChatGLM-4 and other comparative models, was a deliberate mea-
sure to prevent self-evaluation bias. Here are the prompts.
You are serving as a rigorous reviewer and need to score
the following response. Please provide two integer
scores on a 1-5 scale (5 being the best):
# Criteria
Answer Quality (evaluating the overall integrity,
coherence, and fluency of generated answers)
Match Score (assessing the degree of alignment between the
answer and the query's intent, reflecting how
effectively the response addresses the user's
underlying need)
# Output Format (only return JSON)
{
"Answer Quality": <1-5>,
"Match Score": <1-5>,
"Comment": "<concise justification in 50 characters or
less>"
}
# Input
## Question
(User's input query)

AskNearby: LLM-Based Neighborhood Info Retrieval & Recommendations GeoAI ’25, November 3–6, 2025, Minneapolis, MN, USA
Figure 5: Screenshots of the AskNearby system: (a) Recommendation page and (b) Retrieval page.
## Response
(System-generated answer)
B.2 LLM Evaluation Prompt for Community
Life-Circle Recommendations
You are an evaluator for a life-circle recommendation
system.
Your current time is {Time}, and your location is
{Location}.
Based on the recommended post list (see below), output the
following evaluation as JSON:
1. How many posts are relevant to your life circle (within
5 km or a familiar neighborhood)?
2. For each post, give a match score (0.0 to 1.0)
representing how useful the content is to you.
3. At which position (starting from 1) is the post that you
find most interesting?
Output format (only return JSON):
{
"relevant_post_count": <int>,
"match_score": [<float>, <float>, ...],
"top_interest_position": <int>
}
Post Info:
[
{
"sname": "Shenzhen University Town",
"title": "Flea Market | Second-hand Bicycle for Sale",
"post_content": "Selling a nearly new bike on campus,
80% new, sincere offer","time": "2024-03-01 14:52:00",
"latitude": 22.590,
"longitude": 113.943,
"tag": ["Shenzhen", "Nanshan", "Campus", "Second-hand"]
},
...
]
Note: Only return valid JSON. Do not include explanations
or extra text.