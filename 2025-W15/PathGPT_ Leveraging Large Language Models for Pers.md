# PathGPT: Leveraging Large Language Models for Personalized Route Generation

**Authors**: Steeve Cuthbert Marcelyn, Yucen Gao, Yuzhe Zhang, Xiaofeng Gao, Guihai Chen

**Published**: 2025-04-08 09:25:21

**PDF URL**: [http://arxiv.org/pdf/2504.05846v1](http://arxiv.org/pdf/2504.05846v1)

## Abstract
The proliferation of GPS enabled devices has led to the accumulation of a
substantial corpus of historical trajectory data. By leveraging these data for
training machine learning models,researchers have devised novel data-driven
methodologies that address the personalized route recommendation (PRR) problem.
In contrast to conventional algorithms such as Dijkstra shortest path
algorithm,these novel algorithms possess the capacity to discern and learn
patterns within the data,thereby facilitating the generation of more
personalized paths. However,once these models have been trained,their
application is constrained to the generation of routes that align with their
training patterns. This limitation renders them less adaptable to novel
scenarios and the deployment of multiple machine learning models might be
necessary to address new possible scenarios,which can be costly as each model
must be trained separately. Inspired by recent advances in the field of Large
Language Models (LLMs),we leveraged their natural language understanding
capabilities to develop a unified model to solve the PRR problem while being
seamlessly adaptable to new scenarios without additional training. To
accomplish this,we combined the extensive knowledge LLMs acquired during
training with further access to external hand-crafted context
information,similar to RAG (Retrieved Augmented Generation) systems,to enhance
their ability to generate paths according to user-defined requirements.
Extensive experiments on different datasets show a considerable uplift in LLM
performance on the PRR problem.

## Full Text


<!-- PDF content starts -->

PathGPT: Leveraging Large Language Models for
Personalized Route Generation
Steeve Cuthbert Marcelyn1, Yucen Gao1, Yuzhe Zhang1, Xiaofeng Gao1, and
Guihai Chen1
Shanghai Key Laboratory of Scalable Computing and Systems,
Department of Computer Science and Engineering,
Shanghai Jiao Tong University, Shanghai, China
{stephen_5, guo_ke, zhangyuzhe}@sjtu.edu.cn, {gao-xf,
g-chen}@cs.sjtu.edu.cn}
Abstract. The proliferation of GPS-enabled devices has led to the ac-
cumulation of a substantial corpus of historical trajectory data. By lever-
aging these data for training machine learning models, researchers have
devised novel data-driven methodologies that address the personalized
route recommendation (PRR) problem. In contrast to conventional al-
gorithms such as Dijkstra’s shortest path algorithm, these novel algo-
rithmspossessthecapacitytodiscernandlearnpatternswithinthedata,
thereby facilitating the generation of more personalized paths. However,
once these models have been trained, their application is constrained to
thegenerationofroutesthatalignwiththeirtrainingpatterns.Thislimi-
tation renders them less adaptable to novel scenarios and the deployment
of multiple machine learning models might be necessary to address new
possible scenarios, which can be costly as each model must be trained
separately. Inspired by recent advances in the field of Large Language
Models (LLMs), we leveraged their natural language understanding ca-
pabilities to develop a unified model to solve the PRR problem while
being seamlessly adaptable to new scenarios without additional training.
Toaccomplishthis,wecombinedtheextensiveknowledgeLLMsacquired
during training with further access to external hand-crafted context in-
formation, similar to RAG (Retrieved Augmented Generation) systems,
to enhance their ability to generate paths according to user-defined re-
quirements. Extensive experiments on different datasets show a consid-
erable uplift in LLM performance on the PRR problem.
Keywords: Route Recommendation ·Deep Learning ·Large Language
Model ·RetrievalAugmentedGeneration ·NaturalLanguageProcessing.
1 Introduction
1.1 The Personalized Route Recommendation Problem and
Current Challenges
Path planning determines the optimal sequence of vertices for a path between a
givenstartandendpointinaroadnetworkgraph.Ithasmanyapplicationareas,arXiv:2504.05846v1  [cs.IR]  8 Apr 2025

2 Authors Suppressed Due to Excessive Length
including transport, logistics, emergency services, and robotics. Traditionally,
path planning has been viewed as a combinatorial optimisation problem with
well-defined objectives, such as minimising the total travel distance or time. In
recent decades, however, most countries have considerably expanded their road
network infrastructure. While this has created new economic opportunities, the
increased complexity of these road networks has made it more challenging for
individuals to navigate from one location to another without relying on external
systems [1], which led to the massive adoption of GPS-enabled devices. Insight
from the collected trajectories suggested that many users actually adopt paths
thatareneithertheshortestnorthefastest[2][6].Thisdisagreementsuggeststhe
complexity of modelling user intent in path planning through traditional search-
basedframeworkswithexplicitgoals.Themainreasonforthiscomplexityisthat
targets often contain many variables that are not easily modelled in closed form.
For example, users might prioritize scenic paths over shorter ones for leisure
drives [3].
In that case, the usual modelling way can’t satisfy individualized needs.
Hence, there is a need to re-examine the methodologies for path planning to
better align with the complex, real-world objectives observed in practice. In
light of these challenges, several solutions have been proposed such as modeling
the road networks as graphs and directly applying traditional graph algorithms
on the resulting network to find or generate a path, often the shortest, connect-
ing two given points within these road networks. Recently, the proliferation of
GPS-enabled devices has resulted in the accumulation of a vast amount of tra-
jectory data [4], which coupled with the ability of machine learning models to
identify and learn hidden patterns in data has led to the development of new
data-driven algorithms [5] capable of generating paths based on various factors
other than distance such as traffic condition, number of point of interests, etc
which can help in offering a more personalized experience to the user.
However, these models can only generate paths according to a given set
of predetermined constraints or requirements which in most cases cannot be
modified after their training, that is, one model trained to predict a path with
the optimal traffic condition cannot be used to generate one which must go
through a series of given points of interests. In the face of this challenge, one
solution might be training multiple models where each model predicts a single
path tailored to a specific requirement. Although this approach may solve the
original problem, it is quite inefficient since it requires training a model for each
potential requirement that a user could provide, but also infeasible since the
set of all these requirements is unknown before the model’s training, making it
difficult to rely on them to offer a unified solution.
1.2 Our Proposed Solution
Foundation models [7] have been pre-trained on a vast amount of data and
have constantly demonstrated exceptional performance on tasks such as question
answering across various domains, code generation, etc. Inspired by these recent
advances, we postulated that these models, with their extensive knowledge of the

PathGPT: Leveraging LLMs for Personalized Route Generation 3
world and intrinsic capacity to comprehend natural language, could potentially
serve as a unified model capable of generating paths that are more aligned with
the specific requirements of a given user, thus reducing the need for multiple
models and the uncertainty of the provided input query.
Through our experiments, we found that they possess knowledge about the
roadnetworksofsomecitiesand,givenauserquery,theabilitytopredictaroute.
Motivated by these findings, we elected to build a framework, PathGPT, capable
of enhancing the path-generation ability of LLMs by modeling the personalized
route recommendation problem as a natural language processing task.
Specifically, given a database comprised of historical trajectories where each
trajectory is initially a sequence of edge IDs of a given road network, we first
process each path in a way to obtain their corresponding natural language repre-
sentation. Through reverse geocoding, we first obtain a human-readable address
for the source and destination nodes, respectively. The names of the roads tra-
versed by each path are obtained in a similar manner. We then combine this
information to generate a simple text describing the original edges-based path
it relates to, as shown in Figure 1. Then, we store these textual representations
of the historical paths in another external database for further use.
Although prior studies demonstrated that LLMs can comprehend a sequence
of GPS coordinates to a certain extent, our findings revealed that their proba-
bilisticnature andhow inputdata istokenized resultedin lowconsistencyamong
outputs for a given input. Consequently, given LLMs’ proficiency in natural lan-
guage processing tasks, we chose to employ the aforementioned text representa-
tion. Then, for each input prompt to the LLM, the task is to generate a path
given a source, destination, and constraints. We employ a retriever model to ex-
tract paths that may pass through the starting and destination addresses from
the previously created database. In conjunction with the input, these retrieved
historical paths serve as context from which the LLM may extract information
and patterns that may be useful in carrying out the generation task.
Tothebestofourknowledge,PathGPTisthefirstattemptatapplyingLLMs
to the personalized route recommendation problem. Through our experiments,
it has proved to be more than capable of generating routes when the set of con-
straints is unknown before the inference phase, a task that is extremely difficult
for existing models, furthermore, in contrast to black box machine learning mod-
els where it is difficult to fully understand their generation process, PathGPT by
using the text generation capability of LLMs, can provide more insight, which
could be beneficial for users on the verge of starting a new journey.
In summary, the main contributions of this paper are summarized as follows:
–PathGPT converts historical paths from raw edge ID sequences to natural
language descriptions, enabling the LLM to better understand and process
path information. In addition, it further enhances the model’s generalization
capability by generating diverse path types such as fastest path and shortest
path, enabling it to generate more diverse path recommendations.
–We propose PathGPT, which applies an LLM to the PRR problem, leverag-
ing its powerful natural language understanding and reasoning capabilities

4 Authors Suppressed Due to Excessive Length
to generate paths that meet specific user needs. Compared to traditional ma-
chine learning models, PathGPT can handle many different path generation
requirementswithouttheneedtotrainseparatemodelsforeachrequirement.
–To solve the problem of lack of knowledge or hallucination problem that
may occur in the path generation task of the LLM, PathGPT introduces the
retrieval-enhanced generation technique. By retrieving paths similar to the
user query from the historical trajectory database and providing them as
contextual information to the big language model, PathGPT can generate
more accurate and reliable paths.
2 Related Work
2.1 Traditional Path Planning Algorithm
For graph-based path planning problems, Dijkstra and A* are the best-known
traditional approaches [14] [15]. Different algorithms have been derived on this
basis. They are simple and efficient, but also suffer from high complexity. There-
fore, people began to research on improving the algorithms. The parameters
of the A*-DWA algorithm must be set empirically; thus, the algorithm is not
universal [16]. The IA*FC aims to reduce fuel consumption, however, the al-
gorithm still suffers from high computational complexity and long computation
time [17]. NeuroMLR, a learning-based model, achieves better results on spe-
cific datasets but lacks generalization [18]. Traditional machine learning models
have proven their effectiveness. They are usually able to give the shortest path
faster. However, we want our algorithms to have a better generalization and take
into account traffic conditions, fuel consumption, and user-personalized content.
Therefore, we incorporated the LLM into the path-planning algorithm.
2.2 Large Language Models
Recent advances in LLMs have significantly increased their ability to reason
logically and causally [19]. New models like OpenAI GPT, Deepseek [28] are
emerging. This progress benefits from advances in two main areas. First, natu-
ral language understanding allows LLMs to extract relationships between texts,
which ensures LLMs can identify entities, actions, and causal chains [20]. Sec-
ond, transformer architecture empowers combinatorial generalization and sym-
bolic reasoning [21]. Therefore, LLMs possess a greater ability to generalize and
reason. In path planning, the model can reason about user preferences and give
recommendations accordingly.
2.3 Retrieval-Augmented Generation
The knowledge stored in a LLM may be out-of-date [22], and The model some-
times hallucinates [23], which means their answers are irrelevant or incorrect. By
adding professional knowledge, retrieval-augmented models can generate more

PathGPT: Leveraging LLMs for Personalized Route Generation 5
Table 1. List of Symbols and Their Corresponding Meanings.
Symbol Meaning
G Directed graph which represents the road network of a city
V Set of nodes of G
E Set of vertices of G
W′Weights of the edges expressed in terms of edges (road) travel times G
W′′Weights of the edges expressed in terms of edges (road) length G
fdijkstraFunction used to compute Dijkstra shortest path
P A path, sequence of edges or nodes
vsThe start node of any given path P
vdThe destination node of any given path P
D External knowledge database of historical paths
P(i)
hTheith historical path of D
P(i)
fThe generated fastest path related to the ith historical path D
P(i)
sThe generated shortest path related to the ith historical path D
˜P(i)
hThe sequence of road names traversed by P(i)
h
˜P(i)
fThe sequence of road names traversed by P(i)
f
˜P(i)
sThe sequence of road names traversed by P(i)
s
T(i)Combined Textual representation of ˜P(i)
h,˜P(i)
fand ˜P(i)
s
d(i)Embedding vector of T(i)
D′Vector database containing all the embedding vectors d(i)for any i
fencoderFunction that symbolizes the encoder model
q User query
t Retriever instruction
d Embedding vector dimension
m Number of vectors stored in D′which is naturally the number of paths in D
qrAugmented user query
eqrEmbedding vector of qr
M Matrix representing the vector database D′
Dk(q)Top-k retrieved paragraphs with respect to the augmented query qr
ˆP LLM generated path
accurate and reliable results [24]. When we directly tested the effect on LLMs
for route generation, the results were unsatisfactory. We speculate that this is
due to the lack of relevant knowledge in the model. Therefore, we based our
approach on the retrieval-augmented generation(RAG) technique and achieved
significant results.
3 Problem Formulation
In this section, we formally define the problem of personalized route recommen-
dation.

6 Authors Suppressed Due to Excessive Length
Definition 1. (Road network). A road network is a directed graph G= (V,E)
where Vis a set of nodes which denote the road intersections and E ⊆ V × V a
set of edges representing road segments.
Definition 2. (Path). A path or route is an ordered sequence of nodes
P={v1, ..., v k}where each vi∈ Vfori= 1, ..., kandv1, vkcorrespond to the
source and destination nodes respectively. Similarly, a path can also be defined
in terms of edges where P={e1, ..., e k−1}withei∈ Eandei= (vi, vi+1)fori=
1, ..., k−1.
Throughout the rest of this article, we will use the second definition.
Definition 3. (Query). The input query q: (vs, vd, c)is a triple where vsde-
notes the source node, vdthe destination node (here we assume that vs, vd∈ V)
andcis a set of characteristics to describe the path from vstovd.
Definition 4. (Personalized Route Recommendation Problem). Given a road
network G, a query q: (vs, vd, c), we would like to find path P∗that starts from
vsto reach vdwhile satisfying the given user constraint cas much as possible.
4 PathGPT
Human:Generate the 
fastest path from Rua da 
Prelada to Rua de Afonso 
Lopes Vieira.Task: Given a user 
query, retrieve relevant 
passages that answer or 
provide information to 
help answer the query .
Encoder𝑞𝑟=(q,t)
Vector Database D’Top-K 
Prompt
GeneratorHuman :Suppose you are a navigation app like 
google maps and you have been given the 
following historical paths:
Context #1
### Below are three paths that start from Rua da 
Prelada to reach Rua de Afonso Lopes Vieira:
* The fastest path which crosses: Rua da 
Prelada ,…, Rua do Professor Damião Peres, ….
* Then the shortest path which goes through: Rua
da Prelada ,…, Rua de Eugénio de Castro
* And the last but not least is the most used path 
that crosses: Rua da Prelada ,…, Rua de Azevedo 
Coutinho, …
Context #2
-------------------------------------------------------------------
------------
Context # k
-------------------------------------------------------------------
------------
Use that and your knowledge about 
the road network of the city of 
Porto to generate the fastest path 
from Ruada Prelada to Ruade Afonso 
Lopes Vieira. Your answer should 
ONLY include the names of the roads 
traversed by this path and the names 
must be separated by a comma. 𝒆𝒒𝒓𝒆𝒒𝒓,𝑑(1)
……………..
𝒆𝒒𝒓,𝑑(𝑚)𝒆𝒒𝒓,𝑑(𝑗)
……………..
𝒆𝒒𝒓,𝑑(𝑚)
……………..
𝒆𝒒𝒓,𝑑(𝑖)
Ranker
𝒆𝒒𝒓,𝐌=
𝒅(𝟏)
𝒅(𝟐)
…
𝒅(𝐦)
Prompt inputLLM
LLM Response𝐌
=𝒟𝓀𝓆
𝒟𝓀𝓆
Fig. 1.PathGPT framework overview.

PathGPT: Leveraging LLMs for Personalized Route Generation 7
4.1 Framework Overview
To leverage the performance of LLMs on the personalized route recommendation
problem, we adopt a similar method to retrieval augmented generation (RAG)
systems. From an external database comprised of historical trajectories, we re-
trieve those that are the most semantically similar to the user query based on
the provided starting and destination addresses, as well as the characteristics
that the predicted path should satisfy. We first generate a textual representa-
tion of each historical path from the external database as illustrated in Figure
2. Then, by passing each of these representations through an encoder model, we
could compute their corresponding embeddings which are later stored in a vector
database. Given a user query, we use the same encoder to obtain its embeddings
during the inference phase. By calculating the similarity score of each stored
path embedding and the user query, we retrieve the top-k similar paths’ textual
descriptions and additional information. This information is then concatenated
with the query to generate a prompt for the LLM to guide it for the prediction
task, as shown in Figure1. In the following sections, we will delve into the specific
details of each component.
4.2 Data Augmentation and Context Generation
During our initial experiments, we observed that in most scenarios LLMs seemed
to "understand" the path generation problem; however, relying solely on them
without further tuning sometimes led to erroneous results. One of the first chal-
lenges we encountered was about the relative inconsistencies among the answers,
that is, when given exactly similar prompts, the responses from the LLM would
vary to some extent and may even sometimes generate paths that don’t have
anything in common or include locations that are completely unrelated to the
region in question. Thus, we decided to provide additional information to the
initial query by retrieving from a database of historical paths those that cover
the same starting and destination nodes as much as possible and are similar
(e.g., fastest or most commonly used). Taking inspiration from RAG systems,
we hoped this information may serve as a context and reference guide for the
LLM during the generation task.
However, each path Phin this database is initially expressed as a sequence
of edges ID obtained after applying a map-matching algorithm on the corre-
sponding raw trajectory, as shown in Figure 1. Here, each edge ID is assigned
by OpenStreetMap(OSM) [9], a free and editable world map, and serves as an
identifier for each road segment part of the road network. Unfortunately, with-
out context, these edge IDs don’t hold any valuable information for guiding the
LLM during the path generation process. Thus, we cannot use their original
state to create our own context. However, an edge ID can be used as a key to
retrieve important information and details about the corresponding road, such
as its official name, length, etc. This information is stored in the related Open-
StreetMap table and can be retrieved through a simple lookup. And since LLMs
are proficient at understanding natural language, therefore, for each path Phwe

8 Authors Suppressed Due to Excessive Length
(𝑣𝑠,𝑣𝑑)
𝑣𝑠𝑣𝑑𝑓𝑑𝑖𝑗𝑘𝑠𝑡𝑟𝑎
(𝐺,𝑊′𝐺,𝑣𝑠,𝑣𝑑)𝑓𝑑𝑖𝑗𝑘𝑠𝑡𝑟𝑎
(𝐺,𝑊′′𝐺,𝑣𝑠,𝑣𝑑)
Road Network 𝐺
𝑊′𝐺=𝑇𝑟𝑎𝑣𝑒𝑙𝑇𝑖𝑚𝑒𝑠 𝑊′′𝐺=𝐿𝑒𝑛𝑔𝑡ℎ
Fastest path Historical path Shortest path
𝑃ℎ(𝑖)={𝑒1,𝑒2,𝑒3}𝑃𝑓(𝑖)={𝑒′1,𝑒′2,𝑒′3}𝑃ℎ(𝑖)={𝑒1,𝑒2,𝑒3}𝑃𝑠(𝑖)={𝑒′′1,𝑒′′2,𝑒′′3}OSM DATABASE
(𝑃𝑓(𝑖),𝑃ℎ(𝑖),𝑃𝑠(𝑖))
෪𝑃𝑓𝑖={Rua deGonçalo Cristóvã ,…,Rua
Gonçalo Sampaio }
෪𝑃ℎ𝑖={Rua deGonçalo
Cristóvã,…,Avenida da Boavista ,…,Rua
Gonçalo Sampaio }
෪𝑃𝑠𝑖={Rua deGonçalo Cristóvã ,…,Rua
Caldas Xavier,… Rua Gonçalo Sampaio }
Context Generator
There are three paths that start from 
Rua deGonçalo Cristóvã to Rua Gonçalo
Sampaio :
The fastest path which crosses: ….,…..,.
Then the shortest path which goes 
through:…,….,……
And the last but not least is the most 
used path that crosses:….,….,…..Encoder
Vector Database D’𝒅(𝒊)
Fig. 2.The context generation process.
decided to embed some of this information such as the names of the starting
and the destination addresses, the names of the roads traversed, the path type,
etc in a small paragraph of text. Furthermore, in order to offer more variation
and stimulate the model generalization, we generated two additional path types.
This is better illustrated in Figure 2, where for the i-th historical path P(i)
h, we
generate P(i)
fandP(i)
s, which respectively represent the fastest and the shortest
path from vstovd. These two new paths are given by :
P(i)
f=fdijkstra (G, W′
G, vs, vd) (1)
P(i)
s=fdijkstra (G, W′′
G, vs, vd) (2)
While they share the same endpoints as P(i)
h, it is natural to expect some
differences in their respective edge sequences, which could translate into them
traversing different roads. In the next step, we use the edge IDs of P(i)
hand
the newly generated P(i)
fandP(i)
sto retrieve their respective road names and
additional information from the OSM database. With this information and with
the help of a context generator, we create a short paragraph Tdescribing these
three paths in a human-readable manner. This paragraph is then encoded using
an embedding model before being stored in the vector database D′. This can be
expressed as :
d(i)=fencoder (T(˜P(i)
f,˜P(i)
h,˜P(i)
s)) (3)

PathGPT: Leveraging LLMs for Personalized Route Generation 9
4.3 Inference
Since the external database Dand as a result the vector database D′are both
comprised of thousands of items, we must perform a search operation over D′
to find the textual representations of the paths that may cover the starting and
destination addresses provided in the user query. One solution could be to use
lexical search, where the addresses provided by the user could be used as a key
to search through the entire corpus of paragraphs.
Although this solution leads to the correct results, however, given that each
word in the prompt would be compared to every paragraph, it is evident that
thismethodheavilyreliesupontheprompt’sstructure.Specifically,thepositions
within the input query where the words referencing the starting and destination
addresses will appear would be unknown if no directives are provided to the
users on how to format the query. This could be detrimental to their freedom of
expression, which is the main advantage of our framework.
Incontrast,inspiredbyRAG-basedsystems,whichwereintroducedtoextend
or update the knowledge of an LLM following its training phase [10], we chose
to employ semantic search, which enables the retrieval of items based on their
meanings, while affording users the flexibility to construct the prompts in the
manner they deem most suitable. Even though it shares the same end goal with
other enhancing techniques such as fine-tuning, a RAG system, in comparison,
does not modify the weights of the model, nor its abilities to perform other
tasks, instead given a query xand a dataset of documents S, it retrieves from
Sthe document zthat is the most related to xby calculating how similareach
document in Sis tox. To find this document, every document dthat belongs to
Sand the query xfirst pass through a document and query encoder to obtain
their corresponding embeddings (in the original RAG paper the document and
query encoder are the same BERT BASE) model [11]. These are then used as
input to a similarity function to calculate the similarity score between a given
document dand the query xwhere z, the retrieved document, is just the one
associated with the highest similarity score. This can be formally expressed as:
a=exp(d(z),q(x)) (4)
whered(z) =BERT d(z),q(x) =BERT d(q).
In our case, a document dcorresponds to a paragraph T(i)that encapsulates
in natural language all the information retrieved from the OSM database for
given pair (vs, vd), and the paths P(i)
h,P(i)
fandP(i)
sfor which we have already
computed its embedding d(i).
Similarly, using the same encoder model, we can obtain the embedding vector
of the user query qduring the inference phase. However, as indicated in Figure
2 we first concatenate with qa small piece of text tthat will serve as instruction
for the retriever by describing the task we want to achieve. This is necessary
to achieve optimal retrieval since that’s how the encoder was trained, where
omitting it has led to severe performance degradation. From this new augmented
query qr, we then compute its corresponding embedding eqr.

10 Authors Suppressed Due to Excessive Length
Now that we have obtained the embedding vector of the query, and having
previously embedded every paragraph T(i), we then proceed to retrieve from D′
thetop-ksemantically similar T. Roughly put, the idea behind semantic search
is that items whose embeddings are relatively close in the embedding space
should also have a relatively similar meaning. This is explained by the fact that
embedding vectors specify the position of the embedded item in the embedding
space, thus if we introduce the notion of distance and distance functions, we
can thereby calculate the relative distance between two vectors, thus similarity
between two distinct items.
The overall process can be achieved in two stages. In the first stage, we use a
similarityfunction(distancefunction)tocomputethesimilarityscores(distances)
between the augmented query embedding vector eqrand each d(i)stored in D′.
Ifmis the number of embedding vectors stored in D′anddis the dimension
of each vector, the vector database D′can be represented by the matrix M
which dimensions are (m,d)and each of its row is a vector embedding d(i)for
i={1, m}. Therefore the similarity score s(i)ofeqrandd(i)can be computed
by :
s(i) =sim(eqr,Mi) (5)
Where simis the cosine similarity metric given by:
sim(x,y) =x·y
∥x∥∥y∥(6)
andMiis the i-th column of M.
The above computation is then performed on all the rows of M, and the
results are stored in a list S. When using the cosine similarity metric, the more
similar two items are, the lower their similarity scores are, thus we rank all the
elements in Sin increasing order and extract the first kones, given by:
Dk(q)={d(j), ...,d(i)} (7)
where Dk(q)is the set of retrieved textual representations with respect to the
augmented query qrthat will play the role of additional context and reference
guide.
In the next stage, we concatenate Dk(q)with qto generate the final LLM
prompt, as shown in Figure 1.
ˆP=LLM (q|Dk(q)) (8)
For similar reasons to the retriever model, we structure our prompt so that
the first part describes the task in hand to the LLM, then provides the context,
and finally, the question which will result in the generation of a sequence of road
names ˆPas instructed by the prompt.

PathGPT: Leveraging LLMs for Personalized Route Generation 11
Table 2. Datasets details.
Dataset No. of nodes No. of edges No. of trajectories
Beijing 31,199 72,156 1,382,948
Chengdu 3,973 9,255 3,600,503
Harbin 6,598 16,292 1,133,548
Porto 5,330 11,491 1,426,312
5 Experiments
5.1 Experimental Setup
Datasets. We conduct our experiments on publicly available historical taxi tra-
jectory datasets [25] from four different cities, namely Beijing (BJG), Chengdu
(CHG), and Harbin (HRB). In Table 2 we present the details of the datasets.
Initially, each trajectory of any of these datasets is a GPS-sequence; therefore,
the authors [18] applied a map matching algorithm [26] to transform the GPS
sequence to a sequence of edge IDs which we use to create the external knowl-
edge database D. A small portion of the original dataset is also used to prompt
the LLM. To maintain fairness during the evaluation, we exclude it from D.
Ground Truth Generation. Since PathGPT’s principal focus is on the gen-
eration of paths aligned with the user requirements, it is expected that its per-
formance should be judged in that regard. However, due to the personalized
aspect of the problem, it is quite difficult to predict which requirement would be
provided by a user and even more difficult to find a way to test the quality of
the path generated by PathGPT for a wide variety of query types, since every
user may have its own special need.
Thus, we decided to concentrate on some general constraints, such as when
the user may ask for the fastest or shortest path. This was one of the main
objectives of the data augmentation process explained in Section 4.2, where we
useddifferentedgeweights,namelytheedgestraveltimes (W′)andedgeslengths
(W′′)provided by OSM, to generate the fastest and shortest paths, respectively.
Evaluation Metrics We evaluate the accuracy of PathGPT through two pop-
ular metrics: recall and precision, which are given by:
Precision =|P∗∩ˆP|
|P∗|(9)
Recall =|P∗∩ˆP|
|ˆP|(10)
HereP∗is ourground truth. In thescenariowhere the requirement is togenerate
the fastest path we have P∗=P(j)
f, andP∗=P(j)
s, where jrepresents the j-th
sample in the test data.

12 Authors Suppressed Due to Excessive Length
Implementation Details. We use the shortest path algorithm implementation
from Networkx [12] to compute both P(j)
fandP(j)
s. We initially employ the
gte-Qwen2-1.5B-instruct [27] embedding model for encoding both the textual
representation T(i)fori={1, m}and the augmented query qr. As LLM we use
the 14b version of Qwen2.5 [13]. Our experiments are conducted on a virtual
machine running Ubuntu 22.04 LTS with Intel(R) Xeon(R) CPU @ 2.10GHz
and one RTX 4090 GPU card. The source code of PathGPT is available at
https://anonymous.4open.science/r/PathGPT–B423/.
Baselines. We compare the performance of our framework with the following
models:
•CSSRNN [29]: RNN-based approach that models trajectories while taking
into account the topological structure.
•NeuroMLR-D [18]: Trains graph neural network on historical data using
Lipschitz embeddings to generate routes (version of NeuroMLR which uses
Dijkstra’s algorithm).
•NeuroMLR-G (version of NeuroMLR which uses a greedy algorithm) [18]
5.2 Experimental Results
The performance of the above models, including PathGPT, is listed in Table 3
and 4, regarding the fastest and shortest path generation task respectively.
Table 3. Comparison of PathGPT against the baseline models on the precision and
recall metrics on the four datasets for the fastest path generation task.
ModelsPrecision (%) Recall (%)
BJG CHG HRB BJG CHG HRB
CSSRNN 59.5 — 49.8 68.8 — 51.1
NeuroMLR-G 75.6 — 59.6 76.5 — 48.6
NeuroMLR-D 77.9 — 66.1 74 — 49.6
PathGPT 52.3 57.0 48.4 51.3 53.9 37.4
Ablation Study. In this section we examine the effect of the provided context,
i.e. the textual representations of paths extracted from D′during the retrieval
phase, on the performance on the LLM for the PRR problem. We present our
results in Table 5 and Table 6. We observe that for both the generation of the
shortest and the fastest path, the performance of the LLM on both these tasks is
vastly superior to when it is asked to generate either path without any provided
context, proving the effectiveness of our framework.

PathGPT: Leveraging LLMs for Personalized Route Generation 13
Table 4. Comparison of PathGPT against the baseline models on the precision and
recall metrics on the four datasets for the shortest path generation task.
ModelsPrecision (%) Recall (%)
BJG CHG HRB BJG CHG HRB
CSSRNN 59.5 — 49.8 68.8 — 51.1
NeuroMLR-G 75.6 — 59.6 76.5 — 48.6
NeuroMLR-D 77.9 — 66.1 74 — 49.6
PathGPT 46.4 41.4 48.4 48.9 50.3 37.9
Table 5. Comparison between the performance of the LLM with context(PathGPT)
and without on the PRR problem when requirement is set to fastest path generation.
ModelsPrecision (%) Recall (%)
BJG CHG HRB BJG CHG HRB
LLM w/o context 37.2 30.4 33.2 26.8 22.6 20.4
LLM with context (PathGPT) 52.3 57.0 48.0 51.3 53.9 33.7
Influence of context on generated path
6 Discussion
Despite their ability to take into account more than one constraint and the ob-
served increase in the performance of the LLM to generate paths when a specific
type of context is provided, however, there is still a long way and further research
is needed until we see wide adaptation on par with both traditional algorithms
and other existing machine learning solutions, especially in terms of not only
performance but also reliability and scalibility since LLMs are sometimes prone
to hallucination, which during our experiments was manifested by the generation
of paths farther away from the destination.
7 Conclusion
This paper investigates applying LLMs to the personalized route recommenda-
tion problem. Through our experiments, we found that LLms possess the limited
but discernible ability to generate paths according to some constraints provided
by the users using natural language, which, compared to other existing solutions,
were not specifically predetermined and embedded in the model. Thus, they
can freely express which kind of route they are looking for. We then propose
a pipeline, namely PathGPT, capable of enhancing LLM’s ability to generate
personalized paths by providing a context in the form of textual representations
of paths sharing the same endpoint characteristics as the requested path, which

14 Authors Suppressed Due to Excessive Length
Table 6. Comparison between the performance of the LLM with context(PathGPT)
and without on the PRR problem when requirement is set to shortest path generation.
ModelsPrecision (%) Recall (%)
BJG CHG HRB BJG CHG HRB
LLM w/o context 34.3 27.9 29.2 26.0 22.6 20.5
LLM with context (PathGPT) 46.4 41.4 48.4 48.9 50.3 37.9
were obtained through a retriever based on the popular RAG pipeline. Through
further research, we are sure that LLMs will continue to offer new perspectives
on the PRR problem.
Disclosure of Interests. The authors have no competing interests to declare that
are relevant to the content of this paper.
References
1. Zhuojun Jiang, Lei Dong, Lun Wu, Yu Liu, Quantifying navigation complexity
in transportation networks, PNAS Nexus, Volume 1, Issue 3, July 2022, pgac126,
https://doi.org/10.1093/pnasnexus/pgac126
2. Quercia, D., Schifanella, R., Aiello, L.M.: The shortest path to happiness: recom-
mending beautiful, quiet, and happy routes in the city. In: Proceedings of the 25th
ACM Conference on Hypertext and Social Media (HT ’14), pp. 116–125. ACM,
Santiago, Chile (2014). https://doi.org/10.1145/2631775.2631801
3. Ceikute, V., Jensen, C.S.: Routing service quality - local driver behavior versus
routing services. In: 2013 IEEE 14th International Conference on Mobile Data Man-
agement (MDM), vol. 1, pp. 97–106. IEEE Computer Society, Milan, Italy (2013).
https://doi.org/10.1109/MDM.2013.20
4. J. Wang, C. Chen, J. Wu, and Z. Xiong, “No longer sleeping with a bomb: A duet
system for protecting urban safety from dangerous goods,” in Proc. 23rd ACM
SIGKDD Int. Conf. Knowl. Discov. Data Mining, 2017, pp. 1673–1681.
5. Xiucheng Li, Gao Cong, and Yun Cheng. Spatial transition learning on road net-
works with deep probabilistic models. In ICDE, pages 349–360, 2020
6. V. Ceikute andC. S. Jensen, “Routing service quality- local driver behavior versus
routing services,”inMDM,2013,pp.97–106.
7. Brown, T., Mann, B., Ryder, et al., D. Language Models are Few-Shot Learners.
(2020), https://arxiv.org/abs/2005.14165
8. Sarah Hoffmann. Nominatim, 2023. URL https://nominatim.org.
9. OpenStreetMap contributors. OpenStreetMap Foundation, 2024. URL
https://wiki.openstreetmap.org/wiki/Researcher_Information
10. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Wen-Yih, Rocktäschel, T., Riedel, S. & Kiela, D.
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. (2021),
https://arxiv.org/abs/2005.11401
11. Devlin, J., Chang, M., Lee, K. & Toutanova, K. BERT: Pre-training
of Deep Bidirectional Transformers for Language Understanding. (2019),
https://arxiv.org/abs/1810.04805

PathGPT: Leveraging LLMs for Personalized Route Generation 15
12. Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network struc-
ture, dynamics, and function using NetworkX”, in Proceedings of the 7th Python
in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod
Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008
13. Yang, A., Yang, B., Hui, B., et al., Qwen2 Technical Report. ArXiv Preprint
arXiv:2407.10671 . (2024)
14. Delling, D., Goldberg, A.V., Nowatzyk, A., Werneck, R.F.: PHAST: Hardware-
accelerated shortest-path trees. Journal of Parallel and Distributed Computing
73(7), 940–952 (2013).
15. Hart, P.E., Nilsson, N.J., Raphael, B.: A formal basis for the heuristic determina-
tionofminimumcostpaths.IEEETransactionsonSystemsScienceandCybernetics
4(2), 100–107 (1968).
16. Zhong, X., Tian, J., Hu, H., Peng, X.: Hybrid path planning based on safe A*
algorithm and adaptive window approach for mobile robot in large-scale dynamic
environment. Journal of Intelligent & Robotic Systems 99(1), 65–77 (2020).
17. Liu, T., Zhang, J.: An improved path planning algorithm based on fuel consump-
tion. The Journal of Supercomputing, 1–31 (2022).
18. Jain, J., Bagadia, V., Manchanda, S., Ranu, S.: NeuroMLR: Robust and Reliable
Route Recommendation on Road Networks. In: Advances in Neural Information
Processing Systems (NeurIPS), vol. 34, pp. 22070–22082 (2021)
19. Huang, J., & Chang, K. C. (2022). Towards reasoning in large language models: A
survey. arXiv preprint arXiv:2212.10403.
20. Brown, T. B. (2020). Language models are few-shot learners. arXiv preprint
arXiv:2005.14165.
21. Wei,J.,Wang,X.,Schuurmans,D.,Bosma,M.,Xia,F.,Chi,E.,Le,Q.V.,&Zhou,
D. (2022). Chain-of-thought prompting elicits reasoning in large language models.
Advances in Neural Information Processing Systems, 35, 24824–24837.
22. He, H., Zhang, H., Roth, D.: Rethinking with retrieval: Faithful large language
model inference. arXiv preprint arXiv:2301.00303 (2022)
23. Ji,Z.,Lee,N.,Frieske,R.,Yu,T.,Su,D.,Xu,Y.,Ishii,E.,Bang,Y.J.,Madotto,A.,
Fung, P.: Survey of hallucination in natural language generation. ACM Computing
Surveys 55(12), 1–38 (2023)
24. Ren, R., Wang, Y., Qu, Y., Zhao, W.X., Liu, J., Tian, H., Wu, H., Wen, J.R.,
Wang, H.: Investigating the factual knowledge boundary of large language models
with retrieval augmentation. arXiv preprint arXiv:2307.11019 (2023)
25. Jing Lian and Lin Zhang. One-month beijing taxi gps trajectory dataset with taxi
ids and vehicle status. In Proceedings of the First Workshop on Data Acquisition
To Analysis, pages 3–4, 2018.
26. Can Yang and Gyozo Gidofalvi. Fast map matching, an algorithm integrating hid-
den markov model with precomputation. International Journal of Geographical In-
formation Science, 32(3):547–570, 2018.
27. Li, Zehan, Zhang, Xin, Zhang, Yanzhao, Long, Dingkun, Xie, Pengjun, and Zhang,
Meishan. "Towards general text embeddings with multi-stage contrastive learning."
arXiv preprint arXiv:2308.03281 , 2023.
28. DeepSeek-AI, Guo, D., Yang, D., Zhang, et al. DeepSeek-R1: Incentiviz-
ing Reasoning Capability in LLMs via Reinforcement Learning. (2025),
https://arxiv.org/abs/2501.12948
29. Wu, H., Chen, Z., Sun, W., Zheng, B. & Wang, W. Modeling trajectories with
recurrent neural networks. Proceedings Of The 26th International Joint Conference
On Artificial Intelligence . pp. 3083-3090 (2017)