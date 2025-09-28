# From "What to Eat?" to Perfect Recipe: ChefMind's Chain-of-Exploration for Ambiguous User Intent in Recipe Recommendation

**Authors**: Yu Fu, Linyue Cai, Ruoyu Wu, Yong Zhao

**Published**: 2025-09-22 11:35:47

**PDF URL**: [http://arxiv.org/pdf/2509.18226v1](http://arxiv.org/pdf/2509.18226v1)

## Abstract
Personalized recipe recommendation faces challenges in handling fuzzy user
intent, ensuring semantic accuracy, and providing sufficient detail coverage.
We propose ChefMind, a hybrid architecture combining Chain of Exploration
(CoE), Knowledge Graph (KG), Retrieval-Augmented Generation (RAG), and a Large
Language Model (LLM). CoE refines ambiguous queries into structured conditions,
KG offers semantic reasoning and interpretability, RAG supplements contextual
culinary details, and LLM integrates outputs into coherent recommendations. We
evaluate ChefMind on the Xiachufang dataset and manually annotated queries,
comparing it with LLM-only, KG-only, and RAG-only baselines. Results show that
ChefMind achieves superior performance in accuracy, relevance, completeness,
and clarity, with an average score of 8.7 versus 6.4-6.7 for ablation models.
Moreover, it reduces unprocessed queries to 1.6%, demonstrating robustness in
handling fuzzy demands.

## Full Text


<!-- PDF content starts -->

FROM ”WHAT TO EAT?” TO PERFECT RECIPE: CHEFMIND’S CHAIN-OF-EXPLORATION
FOR AMBIGUOUS USER INTENT IN RECIPE RECOMMENDATION
Yu Fu, Linyue Cai, Ruoyu Wu, Yong Zhao*
Sichuan University
Department of Computer Science
Shuangliu District, Chengdu City, Sichuan Province, China
ABSTRACT
Personalized recipe recommendation faces challenges in han-
dling fuzzy user intent, ensuring semantic accuracy, and
providing sufficient detail coverage. We propose ChefMind,
a hybrid architecture combining Chain of Exploration (CoE),
Knowledge Graph (KG), Retrieval-Augmented Generation
(RAG), and a Large Language Model (LLM). CoE refines
ambiguous queries into structured conditions, KG offers
semantic reasoning and interpretability, RAG supplements
contextual culinary details, and LLM integrates outputs into
coherent recommendations. We evaluate ChefMind on the
Xiachufang dataset and manually annotated queries, com-
paring it with LLM-only, KG-only, and RAG-only baselines.
Results show that ChefMind achieves superior performance
in accuracy, relevance, completeness, and clarity, with an av-
erage score of 8.7 versus 6.4–6.7 for ablation models. More-
over, it reduces unprocessed queries to 1.6%, demonstrating
robustness in handling fuzzy demands. These findings vali-
date ChefMind’s effectiveness and feasibility for real-world
deployment.
Index Terms—Recipe Recommendation, Knowledge
Graph, Retrieval-Augmented Generation, Workflow, Large
Language Model
1. INTRODUCTION
With the rapid growth of online recipe content, personal-
ized recipe recommendation systems are gaining increasing
attention[1]. In recent years, various advanced technologie,
such as large language models (LLMs), knowledge graphs
(KGs), and retrieval-augmented generation (RAG), have
been integrated into traditional recommendation frameworks,
enhancing system performance. However, these technolo-
gies each face significant limitations: LLMs may generate
”hallucinations”[2], KGs have limited adaptability in dy-
namic scenarios[3], and RAG heavily relies on retrieval qual-
ity and coverage[4]. More importantly, existing research pri-
marily focuses on improving individual technologies. How
This work is funded by the National Natural Science Foundation of
China (NSFC) under Grant [No.62177007].to organically integrate these methods for complementary
enhancement remains a critical issue demanding in-depth
exploration.
To address these challenges, we propose a novel hybrid
architecture that integrates Chain of Enhancement (CoE),
Knowledge Graph (KG), and Retrieval-Augmented Genera-
tion (RAG) within a unified recipe recommendation system
– ChefMind. The key innovations of our approach are three-
fold: (1) CoE dynamically interprets and refines ambiguous
user queries into structured conditions; (2) KG provides
semantic relational context for accurate and explainable rec-
ommendations; and (3) RAG retrieves real-world culinary
details to enrich response practicality. This paper system-
atically quantifies the contribution of each module through
extensive experiments and validates the overall superiority of
our architecture compared to several baseline methods. The
findings demonstrate the efficacy of combining CoE, KG,
and RAG to achieve a more robust, intuitive, and personal-
ized recipe recommendation system.
2. RELATED WORK
The development of recipe recommendation systems has
transitioned from basic matching to semantic-aware under-
standing. Early systems primarily employed content-based
and collaborative filtering[5, 6]. Content-based methods
used ingredient and nutritional profiles to suggest similar
recipes, offering transparency but struggling to infer im-
plicit preferences[7, 8, 9]. Collaborative filtering analyzed
user interactions-such as ratings and saves-via models like
matrix factorization or neural networks, effectively cap-
turing collective patterns but remaining vulnerable to the
cold-start problem[10, 11, 12]. Hybrid models were subse-
quently proposed to integrate both content and interaction sig-
nals, improving overall recommendation performance[1, 13].
Knowledge graphs (KG) were also introduced to model culi-
nary relationships between ingredients, cuisines, and nutri-
tional attributes[14, 15, 16]. Using graph neural networks,
these systems supported cross-entity reasoning and provided
explainable recommendations, thereby enhancing seman-arXiv:2509.18226v1  [cs.AI]  22 Sep 2025

tic understanding of recipes and dietary contexts[17]. In
recent years, large language models (LLM) and retrieval-
augmented generation (RAG) frameworks have been applied,
where a retrieval component fetches relevant recipes from
large corpora, and the generative language model produces
personalized and context-aware suggestions[18, 19]. This ap-
proach improves performance on ambiguous queries and bal-
ances creativity with accuracy. While some studies suggest
combining knowledge graphs with RAG[20, 21], a deeper
integration of these architectures-particularly within recipe
recommendation-remains an open research area.
3. METHOLOGY
3.1. System Architecture
Fig. 1: The Framework of ChefMind
To address the limitations of traditional recipe recommen-
dation methods, this study proposes ChefMind, a hybrid rec-
ommendation system integrating CoE, KG,RAG, LLM. The
core objectives are verifying effectiveness advantages and ef-
ficiency feasibility, and quantifying individual module contri-
butions using the Xiachufang dataset.
This architecture consists of four complementary modules
forming an end-to-end recommendation loop. The frame-
work of our model is shown by figure 1. CoE acts as the en-
try point for fuzzy demand parsing, converting abstract user
inputs into quantifiable screening conditions. KG enables
semantically accurate recipe screening using structured data
stored in Neo4j. RAG supplements unstructured details by
encoding corpus into 768-dimensional vectors stored in Mil-
vus for similarity retrieval. LLM functions as an ”integrator,”
combining structured KG results and unstructured RAG de-
tails into user-friendly recommendations.
The workflow operates conditionally: for fuzzy demands,
CoE generates refinement logic converted into KG query
conditions; for clear demands, KG directly processes queries.RAG retrieves the most relevant fragments for candidate
recipes, and LLM integrates recipe names with RAG details
to generate final natural language recommendations. Two key
formulas are introduced to formalize critical processes of the
architecture:
3.2. CoE Module
The CoE module serves as the system’s intelligent frontend,
handling user query parsing through five-level progressive
search logic: exact name matching for specific dish queries,
ingredient similarity matching based on available ingredients,
quick home-style dish retrieval, cuisine and flavor matching
for culinary preferences, and broad keyword matching for
comprehensive coverage.
This layered strategy precisely responds to specific queries
while flexibly handling ambiguous requests. Search results
are integrated via matching scores with deduplication ensur-
ing diversity. The CoE module demonstrates its core role in
understanding user intent and translating queries into struc-
tured conditions for knowledge graph retrieval, forming a
complete recommendation loop.
CoE identifies fuzzy demands based on two criteria: the
input contains ambiguous terms or has a length of fewer
than 5 characters. This judgment logic is mathematically
expressed as:
fuzzy(Q) =(
1,if ambiguous terms∈Q∨ |Q|<5
0,otherwise(1)
whereQdenotes the user’s input query. fuzzy(Q) = 1in-
dicates a fuzzy demand (triggering CoE’s 3-step refinement
logic), while fuzzy(Q) = 0indicates a clear demand (directly
entering KG screening).
3.3. KG Module
The KG module provides semantic and structured recipe
data storage and retrieval support, serving as the seman-
tic hub of the recommendation system. Built on the Neo4j
graph database, this module comprises three core node types:
Recipe, Ingredient, and Keyword, forming a rich seman-
tic network through relationships such as ”CONTAINS”
and ”HAS KEYWORD.” Recipe nodes include attributes
like name, dish type, preparation steps, and author; Ingredi-
ent nodes record standardized names, quantities, and units;
Keyword nodes manage tags such as ”home-style” and re-
gional cuisines. Figure 2 illustrates a concrete example of
this knowledge graph structure, showing two interconnected
recipes with their respective ingredients and keywords.
Upon receiving structured query conditions from the CoE
module, the KG module employs multi-hop graph traversal
and semantic matching to retrieve candidate results satisfy-
ing users’ multi-dimensional constraints. Leveraging graph
database advantages, the system supports real-time queries

Fig. 2: Knowledge graph structure example showing two in-
terconnected recipes with their ingredients and keywords
under complex semantic constraints, achieving high-accuracy
retrieval with low latency. The KG module provides com-
plete retrieval paths and associated attributes for each result,
enhancing recommendation interpretability and transparency.
3.4. RAG Module
The RAG module provides dense vector-based recipe re-
trieval, serving as the semantic retrieval engine. Built on
the Milvus vector database, it transforms recipe content into
high-dimensional semantic representations that capture rela-
tionships beyond traditional keyword matching.
Each recipe is encoded into dense vectors using pre-
trained embedding models. The vector database employs
Inner Product metric for similarity computation, achieving
sub-millisecond query latency through approximate nearest
neighbor search.
The module excels at handling fuzzy demands such as
”healthy comfort food” that keyword-based approaches can-
not process effectively. It integrates with the KG module
through hybrid search strategy, where vector-based retrieval
provides broad semantic matching while graph-based re-
trieval ensures structured constraints, significantly enhancing
recommendation accuracy.
RAG retrieves relevant text fragments by calculating vec-
tor similarity using cosine similarity:
Sim(v q,vd) =vq·vd
|vq| · |v d|(2)
wherev qis the 768-dimensional vector of a candidate recipe,
vdis the vector of a text fragment in the RAG corpus, and
higher similarity values indicate greater relevance.
3.5. LLM Module
The LLM module functions as the core connection layer of
ChefMind, integrating structured data from KG and unstruc-
tured details from RAG into coherent recommendations. The
module adopts the DeepSeek model for its balanced perfor-
mance and resource efficiency under GPU constraints.As the connecting component, LLM performs three key
functions: (1) integrating KG’s candidate recipe names and
RAG’s contextual details to avoid fragmented information;
(2) generating natural language output containing recipes and
relevant reasons; (3) adapting expression to demand types by
emphasizing condition matching for fuzzy demands and di-
rect response for clear demands.
The integration process is formalized as:
Rfinal=DeepSeek(R KG,DRAG,P)(3)
whereR KGis KG’s candidate recipes,D RAGis RAG’s de-
tails,Pis the demand-based prompt, andR finalis the inte-
grated recommendation output.
LLM bridges data processing modules (CoE, KG, RAG)
and end users, converting discrete data into natural language
recommendations.
4. EXPERIMENTS AND RESULTS
4.1. Experimental Setup
The core dataset for this experiment is the ”Xiachufang” Chi-
nese recipe dataset. It comprises hundreds of thousands of
authentic user-submitted recipes, including structured and un-
structured information such as dish names, ingredient lists,
detailed cooking instructions, user ratings, and reviews. This
dataset closely aligns with the dietary habits and linguistic
expressions of Chinese users.
To support model evaluation, we constructed a test set of
human-annotated queries, including explicit requests and
fuzzy requests, enabling a comprehensive assessment of
model performance across different scenarios.
4.2. Detailed Introduction to Ablation Models
LLM+KG ModelThe core components of this model are
LLM and KG Its workflow is as follows: it directly receives
user queries, retrieves candidate recipes that meet the condi-
tions through multi-hop graph traversal and semantic match-
ing of Knowledge Graph based on the relationship between
Recipe, Ingredient, and Keyword nodes, and then the Large
Language Model organizes the structured recipe information
returned by Knowledge Graph, including dish names, ingre-
dients, and steps, into natural language recommendations.
LLM+RAG ModelThe core components of this model
are LLM and RAG. Its workflow is as follows: it encodes
user queries and recipe texts intovectors, retrieves the most
relevant unstructured text fragments including step details and
cooking tips through cosine similarity calculation in Milvus,
and the Large Language Model generates recommendations
based on the retrieved results.
ChefMindThe core components of this model are LLM,
CoE, KG and RAG. Its funtion has been mentioned in 3.1

4.3. Scoring Criteria
Scoring Criteria Design The experiment adopts a four-
dimensional quantitative scoring system, with LLM serving
as an objective evaluator to score the recommendation results
generated by the models on a scale of 1 to 10, where 10 is
the highest score. The total score is the average of the four
dimensions, retaining one decimal place. The specific criteria
are as follows:
To evaluate the recommendation results, we designed a
four-dimensional scoring system, where each dimension is
rated on a scale of 1 to 10 and the final score is the average
value:
•Accuracy– measures the degree of consistency be-
tween the recommendation results and the ground-truth
labels. Higher scores indicate better alignment with
user expectations.
•Relevance– assesses how closely the recommenda-
tions match the intent of the user’s query while avoiding
irrelevant or off-topic suggestions.
•Completeness– evaluates the coverage of essential in-
formation, including dish names, ingredients, prepara-
tion steps, and contextual scenarios. Missing items re-
sult in deductions.
•Clarity– reflects the logical structure, readability,
and linguistic standardization of the recommendations.
Poorly ordered steps, inconsistent terms, or chaotic
expressions reduce the score.
4.4. Experimental Result Analysis
The experimental result is that the performance of ChefMind
is significantly better than that of the ablation models (shown
in Table 1. In terms of the overall average total score, Chef-
Mind (8.7 points) is higher than LLM+RAG (6.7 points) and
LLM+KG (6.4 points). ChefMind leads by 2 to 3 points
in the accuracy and relevance dimensions, which reflects the
improvement of Chain of Exploration on intent parsing and
the collaborative value of Knowledge Graph and Retrieval-
Augmented Generation. In terms of the number of unpro-
cessed queries, ChefMind has only 2 unprocessed queries, ac-
counting for 1.6%, which is much lower than LLM+KG (33
unprocessed queries, accounting for 25.6%) and LLM+RAG
(22 unprocessed queries, accounting for 17.1%). Especially
in batches with fuzzy queries, such as batches 3 and 7, Chef-
Mind has only 1 unprocessed query, while LLM+KG has 4 to
5 unprocessed queries.
Table 1: Batch-Level Result Comparison of ChefMind Abla-
tion Experiment (13 Batches, 129 Queries in Total)Batch Total QueriesLLM+KG LLM+RAG ChefMind
Avg Score Unprocessed Queries Avg Score Unprocessed Queries Avg Score Unprocessed Queries
1 10 6.2 3 6.5 2 8.8 0
2 10 6.5 2 6.8 1 8.9 0
3 10 5.8 5 6.0 4 8.2 1
4 10 6.7 2 7.0 1 9.0 0
5 10 6.3 3 6.6 2 8.7 0
6 10 6.9 1 7.2 1 9.1 0
7 10 5.9 4 6.2 3 8.3 1
8 10 6.6 2 6.9 1 8.9 0
9 10 6.4 3 6.7 2 8.6 0
10 10 6.8 1 7.1 1 9.0 0
11 10 6.5 2 6.8 1 8.8 0
12 10 6.1 3 6.4 2 8.5 0
13 9 6.3 2 6.6 1 8.7 0
Overall 129 6.4 33 6.7 22 8.7 2
Note: 1. Avg Score is the average of the four-dimensional scores, retaining
one decimal place; 2. Unprocessed Queries is the number of unprocessed
queries in the batch, unit: piece; 3. Overall is the summary result of 13
batches.
As shown in Table 1, ChefMind consistently outperforms
the ablation models. This superiority is further visualized in
Figure 3, which breaks down the average scores by evaluation
dimension.
Acc RelComp Clarity Overall0246810
6.266.56.8
6.4 6.56.37 7.1
6.78.58.88.698.7
DimensionScore (1-10)
LLM+KG LLM+RAG ChefMind
Fig. 3: Performance comparison across dimensions. Chef-
Mind (mint green) consistently outperforms baselines.
5. CONCLUSION
This paper proposes a hybrid CoE+KG+RAG+LLM architec-
ture, ChefMind, for recipe recommendation in Chinese. Ex-
perimental results demonstrate that the architecture outper-
forms LLM+KG and LLM+RAG models in accuracy, rele-
vance, completeness and clarity, especially in fuzzy demands.
6. REFERENCES
[1] Jill Freyne and Shlomo Berkovsky, “Intelligent food
planning: personalized recipe recommendation,” inPro-
ceedings of the 15th international conference on Intelli-
gent user interfaces, 2010, pp. 321–324.
[2] Chen Ling, Xujiang Zhao, Jiaying Lu, Chengyuan
Deng, Can Zheng, Junxiang Wang, Tanmoy Chowdhury,

Yun Li, Hejie Cui, Xuchao Zhang, et al., “Domain spe-
cialization as the key to make large language models
disruptive: A comprehensive survey,”ACM Computing
Surveys, 2023.
[3] Weiqing Min, Chunlin Liu, Leyi Xu, and Shuqiang
Jiang, “Applications of knowledge graphs for food sci-
ence and industry,”Patterns, vol. 3, no. 5, 2022.
[4] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel,
et al., “Retrieval-augmented generation for knowledge-
intensive nlp tasks,”Advances in neural information
processing systems, vol. 33, pp. 9459–9474, 2020.
[5] Jill Freyne and Shlomo Berkovsky, “Recommending
food: Reasoning on recipes and ingredients,” inInter-
national Conference on User Modeling, Adaptation, and
Personalization. Springer, 2010, pp. 381–386.
[6] Weiqing Min, Shuqiang Jiang, and Ramesh Jain, “Food
recommendation: Framework, existing solutions, and
challenges,”IEEE Transactions on Multimedia, vol. 22,
no. 10, pp. 2659–2671, 2019.
[7] Chia-Jen Lin, Tsung-Ting Kuo, and Shou-De Lin, “A
content-based matrix factorization model for recipe rec-
ommendation,” inPacific-asia conference on knowledge
discovery and data mining. Springer, 2014, pp. 560–
571.
[8] Devis Bianchini, Valeria De Antonellis, Nicola
De Franceschi, and Michele Melchiori, “Prefer: A
prescription-based food recommender system,”Com-
puter Standards & Interfaces, vol. 54, pp. 64–75, 2017.
[9] A Padmavathi and Dipta Sarker, “Recipemate: A food
media recommendation system based on regional raw
ingredients,” in2023 14th International Conference on
Computing Communication and Networking Technolo-
gies (ICCCNT). IEEE, 2023, pp. 1–6.
[10] Shlomo Berkovsky and Jill Freyne, “Group-based
recipe recommendations: analysis of data aggregation
strategies,” inProceedings of the fourth ACM confer-
ence on Recommender systems, 2010, pp. 111–118.
[11] Mansura A Khan, Ellen Rushe, Barry Smyth, and David
Coyle, “Personalized, health-aware recipe recommen-
dation: an ensemble topic modeling based approach,”
arXiv preprint arXiv:1908.00148, 2019.
[12] Mehrdad Rostami, Vahid Farrahi, Sajad Ahmadian,
Seyed Mohammad Jafar Jalali, and Mourad Oussalah,
“A novel healthy and time-aware food recommender
system using attributed community detection,”Expert
Systems with Applications, vol. 221, pp. 119719, 2023.[13] Florian Pecune, Lucile Callebert, and Stacy Marsella,
“A recommender system for healthy and personalized
recipes recommendations.,” inHealthRecSys@ RecSys,
2020, pp. 15–20.
[14] Steven Haussmann, Oshani Seneviratne, Yu Chen,
Yarden Ne’eman, James Codella, Ching-Hua Chen,
Deborah L McGuinness, and Mohammed J Zaki,
“Foodkg: a semantics-driven knowledge graph for food
recommendation,” inInternational Semantic Web Con-
ference. Springer, 2019, pp. 146–162.
[15] Yu Chen, Ananya Subburathinam, Ching-Hua Chen,
and Mohammed J Zaki, “Personalized food recommen-
dation as constrained question answering over a large-
scale food knowledge graph,” inProceedings of the 14th
ACM international conference on web search and data
mining, 2021, pp. 544–552.
[16] Diya Li, Mohammed J Zaki, and Ching-hua Chen,
“Health-guided recipe recommendation over knowledge
graphs,”Journal of Web Semantics, vol. 75, pp. 100743,
2023.
[17] Xiaoyan Gao, Fuli Feng, Heyan Huang, Xian-Ling Mao,
Tian Lan, and Zewen Chi, “Food recommendation with
graph convolutional network,”Information Sciences,
vol. 584, pp. 170–183, 2022.
[18] Fnu Mohbat and Mohammed J Zaki, “Kerl:
Knowledge-enhanced personalized recipe recommen-
dation using large language models,”arXiv preprint
arXiv:2505.14629, 2025.
[19] Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran
Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi, Zhengyuan
Wang, Shizheng Li, Qi Qian, et al., “Searching for
best practices in retrieval-augmented generation,”arXiv
preprint arXiv:2407.01219, 2024.
[20] Linyue Cai, Chaojia Yu, Yongqi Kang, Yu Fu, Heng
Zhang, and Yong Zhao, “Practices, opportunities and
challenges in the fusion of knowledge graphs and large
language models,”Frontiers in Computer Science, vol.
7, pp. 1590632, 2025.
[21] Linyue Cai, Yongqi Kang, Chaojia Yu, Yu Fu, Heng
Zhang, and Yong Zhao, “Bringing two worlds together:
The convergence of large language models and knowl-
edge graphs,” in2024 3rd International Conference
on Automation, Robotics and Computer Engineering
(ICARCE). IEEE, 2024, pp. 207–216.