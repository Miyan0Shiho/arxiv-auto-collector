# Less is More: Denoising Knowledge Graphs For Retrieval Augmented Generation

**Authors**: Yilun Zheng, Dan Yang, Jie Li, Lin Shang, Lihui Chen, Jiahao Xu, Sitao Luan

**Published**: 2025-10-16 03:41:44

**PDF URL**: [http://arxiv.org/pdf/2510.14271v1](http://arxiv.org/pdf/2510.14271v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems enable large language models
(LLMs) instant access to relevant information for the generative process,
demonstrating their superior performance in addressing common LLM challenges
such as hallucination, factual inaccuracy, and the knowledge cutoff.
Graph-based RAG further extends this paradigm by incorporating knowledge graphs
(KGs) to leverage rich, structured connections for more precise and inferential
responses. A critical challenge, however, is that most Graph-based RAG systems
rely on LLMs for automated KG construction, often yielding noisy KGs with
redundant entities and unreliable relationships. This noise degrades retrieval
and generation performance while also increasing computational cost. Crucially,
current research does not comprehensively address the denoising problem for
LLM-generated KGs. In this paper, we introduce DEnoised knowledge Graphs for
Retrieval Augmented Generation (DEG-RAG), a framework that addresses these
challenges through: (1) entity resolution, which eliminates redundant entities,
and (2) triple reflection, which removes erroneous relations. Together, these
techniques yield more compact, higher-quality KGs that significantly outperform
their unprocessed counterparts. Beyond the methods, we conduct a systematic
evaluation of entity resolution for LLM-generated KGs, examining different
blocking strategies, embedding choices, similarity metrics, and entity merging
techniques. To the best of our knowledge, this is the first comprehensive
exploration of entity resolution in LLM-generated KGs. Our experiments
demonstrate that this straightforward approach not only drastically reduces
graph size but also consistently improves question answering performance across
diverse popular Graph-based RAG variants.

## Full Text


<!-- PDF content starts -->

Preprint.
LESS ISMORE: DENOISINGKNOWLEDGEGRAPHS
FORRETRIEVALAUGMENTEDGENERATION
Yilun Zheng1∗, Dan Yang1∗, Jie Li1, Lin Shang2, Lihui Chen1, Jiahao Xu1, Sitao Luan3,4†
1Nanyang Technological University,2Nanjing University,3Mila, Quebec AI Institute,
4University of Montreal
Email: yilun001@e.ntu.edu.sg, luansito@mila.quebec
ABSTRACT
Retrieval-Augmented Generation (RAG) systems enable large language models
(LLMs) instant access to relevant information for the generative process, demon-
strating their superior performance in addressing common LLM challenges such
as hallucination, factual inaccuracy, and the knowledge cutoff. Graph-based RAG
further extends this paradigm by incorporating knowledge graphs (KGs) to lever-
age rich, structured connections for more precise and inferential responses. A
critical challenge, however, is that most Graph-based RAG systems rely on LLMs
for automated KG construction, often yielding noisy KGs with redundant en-
tities and unreliable relationships. The redundancy not only degrades retrieval
and generation performance but also increases computational cost. Crucially, cur-
rent research does not comprehensively address the denoising problem for LLM-
generated KGs. In this paper, we introduce DEnoised knowledge Graphs for
Retrieval Augmented Generation (DEG-RAG), a framework that addresses these
challenges through: (1) entity resolution, which eliminates redundant entities, and
(2) triple reflection, which removes erroneous relations. Together, these tech-
niques yield more compact, higher-quality KGs that significantly outperform their
unprocessed counterparts. Beyond these methods, we conduct a systematic eval-
uation of entity resolution for LLM-generated KGs, examining different blocking
strategies, embedding choices, similarity metrics, and entity merging techniques.
To the best of our knowledge, this is the first comprehensive exploration of en-
tity resolution in LLM-generated KGs. Our experiments demonstrate that this
straightforward approach not only drastically reduces graph size but also consis-
tently improves question answering performance across diverse popular Graph-
based RAG variants. Code is available at (https://github.com/157114/Denoise).
1 INTRODUCTION
Large Language Models (LLMs) have made significant progress in natural language processing,
understanding and reasoning (Zhao et al., 2023; Jin et al., 2025). However, their capabilities are
limited by the delayed access to up-to-date information, susceptibility to hallucination, and weak
long-term memory (Zhao et al., 2023; Huang et al., 2025; Wang et al., 2023). To mitigate these
issues, Retrieval-Augmented Generation (RAG) (Lewis et al., 2020) has emerged to ground LLMs
with external knowledge. Given a user query, a RAG system retrieves relevant information from a
knowledge base, augments the query with the retrieved context, and then generates a response. RAG
enables LLMs to access updated facts, and rapidly adapt to new domain knowledge.
Traditional RAG systems (Karpukhin et al., 2020) retrieve isolated text chunks and ignore relation-
ships among them, which weakens multi-hop reasoning (Yang et al., 2018) and overall coherence
(Siriwardhana et al., 2023). Graph-based RAG (Edge et al., 2024; Guo et al., 2024; Jimenez Gutier-
rez et al., 2024) addresses it by structuring knowledge as a graph and retrieving over that structure.
Connectivity among entities allows models to consider inter-document relations rather than treating
units as independent chunks, enabling fine-grained, relation-aware retrieval (Hong et al., 2025).
∗Equal contribution.
†Corresponding author.
1arXiv:2510.14271v1  [cs.CL]  16 Oct 2025

Preprint.
Form changes
MultilingualAbbrevationSynonym
TypoCase sensitivityLLMsllms
LLLMsLarge Language 
Models
Pretrained
Language ModelsLLM
modelos de 
lenguaje grandes
Figure 1: Redundant concept synonyms for “LLMs” in a knowledge graph. Orange dashed lines
indicate synonymic equivalences showing why these entities convey the same meaning as “LLMs”.
As we all know, the quality of graph is critical to the success of graph mining (Xue & Zou, 2022;
Luan et al., 2024; Zheng et al., 2025), and many graph-based RAG systems focus on constructing
knowledge graphs (KGs) from corpora with LLMs. However, the resulting graphs are often noisy
and redundant (Huang et al., 2024a). During entity and relation extraction, unlike human experts
who can accurately recall and connect new concepts to previously identified entities, LLMs often
struggle to consistently maintain earlier entities and relations due to limited long-context capabili-
ties, which leads to duplicates (Lairgi et al., 2024). As illustrated in Figure 1, the extracted entity
“LLMs” may co-occur with its variants that represent the same concept,e.g.,“LLM” (morphology),
“llms” (casing), “modelos de lenguaje grandes” (multilingual), and “Large Language Models” (ab-
breviation expansion). Existing methods, including LightRAG (Guo et al., 2024), MS GraphRAG
1(Edge et al., 2024), and HippoRAG (Jimenez Gutierrez et al., 2024), typically rely on string-
matching heuristics to merge similar entities, leaving many duplicates unresolved. Theseredun-
dant entitiesinflate storage, degrade retrieval efficiency and precision. Besides, some outdated and
incorrect facts in external corpora(Rietveld et al., 2004; Feng et al., 2025; Mo ¨ell & Sand Aronsson,
2025) yielderroneous triplesin LLM-generated graphs, which mislead retrieval and generation.
To simultaneously reduce the size and improve the quality of generated graphs, we propose DE-
noised knowledge Graphs for Retrieval Augmented Generation (DEG-RAG), which takes entity res-
olution to remove redundancy, and triple reflection to filter erroneous relations in LLM-generated
knowledge graphs for RAG. Entity resolution identifies and links records that refer to the same en-
tity (Ebraheem et al., 2017) and is widely used in traditional KG consolidation (Berrendorf et al.,
2020). We conducted comprehensive evaluation and studies tailored to Graph-based RAG, spanning
different blocking, entity-embedding, matching, and merging strategies.
Our experiments show that, while removing40%of the entities and relations in LLM-generated
KGs, DEG-RAGconsistently improves the performance of four representative Graph-based RAG
approaches, underscoring the importance of KG quality over its size. We further study the design of
different components comprehensively and come up with some interesting findings,e.g.,type-aware
blocking is the most effective blocking method, traditional KG embeddings can rival LLM em-
beddings, neighborhood-based similarity sometimes outperforms ego-based measures, and simple
merging often surpasses synonym-edge addition. Together, these findings offer practical guidance
for constructing high-quality LLM-generated KGs and for developing more efficient and accurate
Graph-based RAG systems, with potential extensions to a wide range of KG-based LLM applica-
tions (Choudhary & Reddy, 2023; Wang et al., 2025; Wang, 2025). In summary, our contributions
are as follows:
• We propose DEG-RAG, which leverages entity resolution and triple reflection to reduce
graph size while improving KG quality for better Graph-based RAG.
• To the best of our knowledge, we are the first to conduct a comprehensive study of en-
tity resolution for Graph-based RAG, implementing and evaluating different components,
including blocking, entity-embedding, matching, and merging strategies.
• Our experiments demonstrate that DEG-RAGimproves the performance of four graph-
based RAG methods across four benchmark QA datasets by removing approximately40%
1To avoid ambiguity, we use MS GraphRAG to refer to the specific GraphRAG method proposed in (Edge
et al., 2024), and Graph-based RAG to refer to the general class of approaches that leverage knowledge graphs.
2

Preprint.
of entities and relations. We further analyze how different components of entity resolution
contribute to Graph-based RAG performance.
2 RELATEDWORK
Retrieval Augmented Generation (RAG) enables Large Language Models (LLMs) to utilize updated
information (Su et al., 2024), access domain-specific knowledge (Zhang et al., 2024), and reduce
hallucinations (Huang et al., 2025). Traditional RAG systems (Karpukhin et al., 2020) organize
external knowledge as isolated database chunks, which limits performance in complex reasoning
(Yang et al., 2018; Jiang et al., 2024) and contextual completeness (Lu et al., 2025; Zhong et al.,
2025). To address these limitations, Graph-based RAG presents external information as graphs,
retrieving relevant data by considering inter-relationships (Peng et al., 2024). MS GraphRAG (Edge
et al., 2024) constructs communities and generates answers based on community summaries, while
LightRAG (Guo et al., 2024) retrieves relevant entities, relationships, and subgraphs using keywords
from queries. HippoRAG (Jimenez Gutierrez et al., 2024) employs PageRank (Page et al., 1998)
for efficient entity retrieval. KAG (Liang et al., 2024) integrates knowledge graphs (KGs) with
LLMs through logical-form-guided reasoning, knowledge alignment, and fine-tuning. Despite these
advancements, the quality of LLM-generated KGs remains a challenge, as they are often redundant
and noisy, hindering efficient knowledge storage and high-quality generation (Zhou et al., 2025).
Entity resolution, which links data records referring to the same real-world entity, is crucial for con-
structing high-quality KGs (Pujara & Getoor, 2016; Obraczka et al., 2021). Existing approaches fall
into three categories: (1) Traditional methods use string similarity (Yu et al., 2016; Papadakis et al.,
2023), heuristic rules (Abu Ahmad & Wang, 2018; Lee et al., 2013), or manually designed schemas
(Efthymiou et al., 2019) to identify equivalent entities. These methods are computationally efficient
and interpretable but struggle with noisy, incomplete, or multilingual data. (2) Embedding-based
methods represent entities in continuous vector spaces, matching based on representation similarity.
This includes LLM-based embeddings (Li et al., 2020) and KG embeddings like TransE (Bordes
et al., 2013), DistMult (Yang et al., 2014), and ComplEx (Trouillon et al., 2016), as well as Graph
Neural Networks (GNNs)-based approaches (Schlichtkrull et al., 2018). These techniques capture
structural dependencies across graphs, offering robustness over heuristic methods. (3) LLM-based
methods leverage LLMs through prompting (Peeters et al., 2023) or fine-tuning (Steiner et al., 2025)
to identify semantically equivalent entities, providing strong generalization capabilities, though they
require careful design for scalability and reliability.
Although many entity resolution methods exist, few focus on improving LLM-generated KG qual-
ity. For example, MS GraphRAG (Edge et al., 2024) and LightRAG (Guo et al., 2024) use simple
string matching for duplicate entity identification. HippoRAG (Jimenez Gutierrez et al., 2024) intro-
duces synonym relations based on cosine similarity, and KAG (Liang et al., 2024) predicts synonym
relations from one-hop neighbors, merging entities accordingly. However, the impact of enhancing
KG quality on Graph-based RAG is largely unexplored. This paper systematically investigates how
different entity resolution methods affect the performance of Graph-based RAG, alongside triple
reflection, contributing uniquely beyond previous studies.
3 PRELIMINARIES
In this section, we introduce the notations and the process of Graph-based RAG. Given a set of
external documentsD= [d 1, d2, . . . , d N], Graph-based RAG constructs a knowledge graph (KG)
G= (E,R,T,A), whereE,R, andTdenote the sets of entities, relation types and triples, andA
represents the textual description for each entity. The neighbors of an entitye∈ Eare defined as the
set of entitiesN(e)that are directly connected toethrough relationr∈ R:
N(e) ={e′∈ E |(e, r, e′)∈ T ∨(e′, r, e)∈ T, r∈ R}.(1)
Then, given a user queryQ, the RAG system (1) retrieves relevant contents fromGvia a retrieval
functionR(·), (2) augments the queryQwith retrieved context using an augmentation function
Aug(·), and (3) generates the final answerYwith LLMsM. Formally:
Y=M ◦Aug
Q,R(Q,G)
.(2)
Specifically, the raw documentsDare first segmented into text chunksC= [c 1, c2, . . . , c M]. For
each chunkc m∈ C, a LLM-based named-entity recognition functionM NER(·)is applied, leads to
3

Preprint.
a set of raw triples, entities, and relations:
Tm=M NER(cm),T=SM
m=1Tm,E={e 1, e2|(e1, r, e 2)∈ T },R={r|(e 1, r, e 2)∈ T }. (3)
where each entitye∈ Ecarries its local textual contextA(e). Here, the LLM extractedEmay con-
tain duplicates, aliases, or simple variations. To construct a coherent KG, a deduplication function
ϕ:E 7→ E∗is applied, which maps each raw entity to a unique canonical entityϕ(e). Then we have
the revised entity, triple, and relation sets as:
E∗={ϕ(e)|e∈ E},T∗={(e 1, r, e 2)|(e 1, r, e 2)∈ T, e 1∈ E∗, e2∈ E∗},R∗={r|(e 1, r, e 2)∈ T∗}(4)
For each canonical entitye∗∈ E∗, we aggregate the textual description with a merge operator⊕:
A∗(e∗) =M
{ei:ϕ(e i)=e∗}A(ei)(5)
The final denoised KG isG∗= (E∗,R∗,T∗,A∗), enabling more efficient retrieval.
4 DENOISINGKNOWLEDGEGRAPHS
In most popular Graph-based RAG systems, such as LightRAG (Guo et al., 2024) and MS
GraphRAG (Edge et al., 2024), a simple string matching strategy is used as the deduplication func-
tion to denoise KGs. However, in this way, entities with the same semantic meaning but different
forms,e.g.,case sensitivity, abbreviation, synonym, multilingual, and typos, will be missed and iso-
lated from each other. This will lead to a coarse and redundant KG that impedes efficient storage
and retrieval in Graph-based RAG systems. To enhance the performance of Graph-based RAG by
denoising LLM-generated KGs, we propose to remove redundant entities by entity resolution in Sec-
tion 4.1 and remove unreasonable edges by triple reflection in Section 4.2. This framework enhances
the quality of the KGs while reducing their size.
4.1 ENTITYRESOLUTION
Entity resolution for KGs involves several key steps(Christophides et al., 2020), (1)Blocking:par-
titions raw entities into blocks to minimize the number of entity pairs that need to be compared. (2)
Matching and Grouping:identify entities that represent the same real-world object and then put
these matched entities into groups representing a single resolved entity. (3)Merging and Linking:
combine the raw entities in each cluster into a canonical representation and update the KG by creat-
ing or deleting relations as needed. With the above steps, we introduce how to use entity resolution
to improve the quality of LLM-generated KGs as follows.
Market Price
of StockStock Price
Stock Holder
Blocking Entity setInvestor
behaviors
Covid-19
Coronavirus
Disease 2019Healthcare
Center
Market Price
of StockMarket Price
of Stock
Worldwide
Market PriceStock Price
Stock PriceMarket Price
of Stock
Name: Market Price... 
Description: how the 
market price was 
determined, but this 
is a later ...
Name: Stock Price
Description: 
how the market price was determined, but this is 
a later ...
This price is not fixed and fluctuates constantly 
throughout the trading ...Name: Stock Price 
Description: This 
price is not fixed and 
fluctuates constantly 
throughout the 
trading ...Stock
holderNVIDIA Analyst
Reports
Stockholder
NVIDIAAnalyst ReportsStock Price
Stock PriceStock HolderInvestor
behaviors
Covid-19Covid-19
Coronavirus
Disease 2019Coronavirus
Disease 2019
Healthcare
Center
Matching & Grouping Merging & Linking⊕
Figure 2: The overall framework of entity resolution for knowledge graphs (Christophides et al.,
2020).
Blocking.To reduce computational costs and unnecessary entity comparisons, blocking is applied
to the entity setEbefore entity matching (Papadakis et al., 2019). Formally, blocking is a mapping
Block:E 7→ B={B 1, B2, . . . , B K},K[
k=1Bk=E(6)
4

Preprint.
where each blockB kis a subset of entities that are more likely to be matched. In this paper, we
consider three types of blocking strategies: semantic-based, entity type-based, and structural-based
(Christophides et al., 2020).
(1) Semantic-Based Blocking. Entities are represented as embeddings generated from their descrip-
tionsA(e)using an embedding modelf emb(·). The entity set is partitioned intokclusters by:
B=kmeans 
{femb(A(e))|e∈ E}, k
,
To avoid manual selection of cluster numberk, we use a rule-of-thumb heuristick=q
|E|
10(Yuan
& Raubal, 2012). This strategy leverages global semantic similarity but is computationally more
expensive for large graphs.
(2) Entity Type-Based Blocking. Entities are first classified into types using a type mapping function
τ:E 7→Ω. Entities with the same typet∈Ωare grouped into the same block:
B={{e∈ E |τ(e) =t} |t∈Ω}.
If a block contains too many entities, we further subdivide it usingk-means. The entity type-based
blocking limits the matches within the same type of entities, which avoids excessive pair compar-
isons.
(3) Structural-based Blocking. This strategy exploits graph connectivity under the assumption that
semantically similar entities are likely to share neighbors. If an entityehas at least two neighbors,
we construct a block for its neighbor setN(e), and the set of final structural-based blocks is then
B={N(e)|e∈ E,|N(e)| ≥2}
This blocking is based on the assumption that entities co-occur as neighbors of the same nodes are
more likely to present the same meaning,e.g.,“Large Language Models” and “Pretrained Language
Models” may be placed in the same block if they both connect to the entity “GPU” through the
relation “run on.”. Therefore, the structural context of shared neighbors serves as a strong signal for
blocking.
Matching and Grouping.After blocking, the objective is to identify sets of entities in each block
that represent the same concept then group entities with the same meaning. Given a blockB⊆ E,
the matching function derives a partition:
Match:B7→G={G 1, G2, . . . , G L},L[
l=1Gl⊆B,(7)
where eachG lis a group of equivalent entities. To match entities, we first obtain the embedding of
each entityh(e)in the KG, then select the entity embedding for matching. Specifically, embedding
methods used in this paper include KG embeddings: TransE (Bordes et al., 2013), DistMult (Yang
et al., 2014), and ComplEx (Trouillon et al., 2016); graph neural network embeddings: CompGCN
(Vashishth et al., 2019) and R-GCN (Schlichtkrull et al., 2018); and LLM embeddings of Qwen3-
Embedding-8B (Zhang et al., 2025).
To match similar nodes with proper information after embedding, we consider the calculation of
the following similarity scores: (1)Ego node similarity.It compares entity embeddingsh(e i)and
h(ej), which is computationally efficient but may miss structural context. (2)Neighbor similarity.
It compares averaged neighbor embeddings ¯hN(ei)and ¯hN(ej), leveraging structural context to
identify entities with similar roles. (3)Type-aware Neighbor similarity.It compares type-specific
averaged neighbor embeddings ¯hNt(ei)and ¯hNt(ej)for each typet∈Ω, whereN t(e) ={e′∈
N(e)|τ(e′) =t}, then averages across types: sim(e i, ej) =1
|Ω|P
t∈Ωsimt(¯hNt(ei),¯hNt(ej)).
This reduces noises from irrelevant neighbors and enables precise matching within specific entity
types, particularly when entities of different types may have fundamentally different embedding
distributions. (4)Ego+neighbor similarity.It considers both the ego node and neighbor information
by concatenating the embeddings in (1) and (2). (5)Ego+Type-aware neighbor similarity.It
considers both the ego node and subset of neighbor information by concatenating the embeddings
used in (1) and (3). Each matching method captures different aspects of entity similarity and presents
distinct trade-offs.
5

Preprint.
After matching, entitiese iande jare grouped together if their similarity exceeds thresholdδ ER, and
we assign each entity to a group using the functiong:E 7→G.
Merging or Linking.Once entity groupsGare obtained, we finalize the KGG∗by editing the
previous KGs with the following three strategies:
(1) Direct Merging. This approach first selects a single canonical entitye∗
l=ϕ(G l)given a group
Gl, whereϕ(·)refers to a canonical selection function. In this paper, we use random selection
forϕ(·). Then, all the other entities inside the groupG lare merged into the canonical entityˆe l.
The KG is updated by appending the descriptions of merged entities to that of the canonical entity,
reconnecting their relations to the canonical entity, and removing relations that involve the merged
entities. The above process can be expressed as:
E∗={ϕ(G l)|G l∈G},A∗(ϕ(G l)) =[
e∈G lA(e),∀G l∈G(8)
T∗={ϕ(g(e 1)), r, ϕ(g(e 2))|(e 1, r, e 2)∈ T, ϕ(g(e 1))̸=ϕ(g(e 2))}.(9)
If the merged description of a canonical entity becomes too long, we summarize it to prevent overly
long inputs from a single entity during retrieval. The merge of similar entities effectively reduces the
storage cost. However, because numerous modifications are made to the original entity and relation
sets, the quality of the resulting knowledge graph largely depends on the effectiveness of the entity
embedding or matching methods used.
(2) Synonym Linking Only. This approach add a synonym relationr synbetween merged entitye′
and canonical entityϕ(G l)inside each groupG lwithout the modification of entity set and attributes,
which can be described as:
T∗=T ∪ {(e′, rsyn, ϕ(G l))|e′∈Gl\ϕ(G l), Gl∈G ent}.(10)
This method keeps the minimal changes to the original KGG, yet still cannot well resolve dupli-
cation of conceptually similar entities insideG, leading to redundancy and low-efficiency during
retrieval.
(3) Merging with Synonym Linking. To prevent the information loss of merged entities as in directly
merging, inside each groupG l, this approach merges attributes and relations to the canonical entity
ϕ(Gl)first, then adds synonym relationsr syntowards canonical entityϕ(G l). In this case, the entity
setEremains unchanged, the relation setRis updated by Equation (9), then Equation (10), and the
attributes is updated by Equation (8).
4.2 TRIPLEREFLECTION
Since the external information in the documents may contain erroneous content, the triples extracted
by LLMs are not always trustworthy (Huang et al., 2024b; Han et al., 2023). Besides, due to the
batched generation of name-entity recognition of chunks, errors may also occur (Lu et al., 2024).
Therefore, we use LLM-as-judge to remove the low-quality triple. Specifically, given a triple,
composed of source entity, relation, and target entity, we let LLM to predict a reliability score
s=M judge(e1, r, e 2). Then, we filter out the triples that are below a thresholdδ TRand the final
relation set that we obtain is
T∗={(e 1, r, e 2)|(e 1, r, e 2)∈ T,M judge(e1, r, e 2)≥δ TR}(11)
4.3 ANALYSIS
Under the construction of KGs in Section 3, if no entity resolution is applied,i.e.,the deduplication
function becomes identity function, yielding a union of subgraphs with no cross edges. Retrieval
over such a disconnected graph reduces to selecting the information of independent triples that a
vanilla retriever would select. Formally, we summarize the claim in Proposition 1 as below, where
the proof is provided in Appendix D.
Proposition 1.Given a graph-based RAG and a vanilla RAG system that share the same augmen-
tation and generation processes, the absence of entity resolution causes the graph-based RAG to
degrade into vanilla RAG.
Proposition 1 demonstrates that any benefit of Graph-based RAG over vanilla RAG necessarily
comes from the connectivity created by entity resolution.
6

Preprint.
5 EXPERIMENTS
In this section, we comprehensively evaluate the effectiveness the denoising approach mentioned
in the previous section for Graph-based RAG systems. We first introduce the experimental settings
in Section 5.1. Then, we demonstrate that entity resolution can significantly reduce the scale of
the original graph while improving question-answering performance on Graph-based RAG systems
in Section 5.2. In Section 5.3, we test and analyze how different components in entity resolution
influence the overall performance. we study the impact of entity reduction ratio and relation reduc-
tion ratio on the performance of Graph-based RAG in Section 5.4. Then, we conduct an ablation
study in Section 5.5 to evaluate the impact of different deletion methods and LLM API. Additional,
we conduct a detailed case study in Appendix B.3 to illustrate the qualitative differences between
knowledge graphs before and after the denoising process.
5.1 EXPERIMENTALSETUP
Datasets and metricsWe evaluate the performance of Graph-based RAG on four datasets from
UltraDomain benchmark (Qian et al., 2025) following (Guo et al., 2024), includingAgriculture, CS,
Legal, andMix.Agriculture, CS,andLegalcontains domain-specific knowledge, whileMixincludes
a broad spectrum of disciplines. Please refer to Appendix A for details of data statistics. Different
Graph-based RAG systems are tested by question-answering tasks. We use an LLM as a judge to
conduct pairwise comparisons between the responses of two methods, where a winning rate greater
than50%indicates that one method outperforms the other, and vice versa. The evaluation considers
four dimensions: comprehensiveness, diversity, empowerment, and overall quality. The detailed
evaluation process is shown in Appendix C.5.
BaselinesWe select four popular Graph-based RAG methods as our baselines: (1) LightRAG (Guo
et al., 2024). (2) HippoRAG (Jimenez Gutierrez et al., 2024). (3) LGraphRAG (Edge et al., 2024).
(4) GGraphRAG (Edge et al., 2024).
Implementation details.We implement our experiment based on DIGIMON (Zhou et al., 2025),
which is a framework that stably implements many variants of Graph-based RAG and provide a fair
and unified comparison among these methods. For efficient indexing and retrieval, the entities and
relations are stored in vector dataset bases implemented by Llama Index (Liu, 2022). We use open
sourced Qwen3-235B-A22B-Instruct-2507 (Team, 2025) for the LLM API calling, which natively
supports 256K context. The model is deployed using VLLM (Kwon et al., 2023) on a Linux server
with 8 H20 GPUs. We use Qwen3-Embedding-8B (Zhang et al., 2025) as the embedding model
during index building and semantic blocking. For the KG embedding, we use pykeen (Ali et al.,
2021), which is design for many types of KG embedding. By default, we set the entity reduction
ratio as40%of the total size of the entity set,δ TRof triple reflection as0.2, semantic-based method
for blocking, LLM embeddings for entity embedding, ego-based similarity for matching, direct
merging in merging step. Please refer to Appendix C for more implementation details.
5.2 IMPACT OFKNOWLEDGEGRAPHDENOISING
To validate the effectiveness of our proposed DEG-RAG, we compare the performance of baseline
Graph-based RAG with denoised KGs and original KGs on four datasets. As shown in Table 5.1,
after reducing40%of the entities and removing erroneous relations, the performance of Graph-based
RAG on cleaned KGs is better than the original KGs in most cases. This indicates the necessity of
denoising KGs for Graph-based RAG. Note that for HippoRAG, the performance is not significantly
improved on theLegalandMixdatasets. This is because the entity set of the KG in HippoRAG only
contains entity names without descriptions, limiting the performance of entity resolution.
5.3 COMPONENTANALYSIS OFENTITYRESOLUTION
We further study the impact of different components of entity resolution on the performance of
Graph-based RAG. Figure 3 shows the winning rate averaged across four metrics (Comprehensive,
Diversity, Empowerment, and Overall) on denoised KGs with different components of blocking
type, entity embedding, similarity mode, and merge type. We find that: (1) Entity type-based block-
ing is more effective than semantic-based or structure-based blocking. We speculate that entity type
is a better and more natural inductive bias for entity resolution and can lead to more robust denoised
graph, which is important for graph mining (Luan et al., 2022; Zheng et al., 2024). (2) Traditional
KG embeddings can rival LLM embeddings. In theLegalandAgriculturedatasets, LLM embed-
dings underperform ComplEx embeddings (Trouillon et al., 2016), which represents entities and
7

Preprint.
Table 1: Performance comparison of graph-based RAG methods on original and cleaned knowledge
graphs across four datasets. The evaluation is based on winning rates by comparing responses
generated from original versus cleaned knowledge graphs.
Dataset DimensionLightRAG HippoRAG LGraphRAG GGraphRAG
Orig. Clean Orig. Clean Orig. Clean Orig. Clean
AgricultureComprehensive 43.60%56.40%49.80%50.20%48.80%51.20%47.79%52.21%
Diversity 41.60%58.40%43.78%56.22%40.00%60.00%36.14%63.86%
Empowerment 42.00%58.00%47.39%52.61%45.60%54.40%47.79%52.21%
Overall 42.40%57.60%48.19%51.81%47.20%52.80%47.39%52.61%
CSComprehensive 39.20%60.80%49.17%50.83%47.18%52.82%48.19%51.81%
Diversity 40.00%60.00%35.54%64.46%43.55%56.45%44.58%55.42%
Empowerment 40.80%59.20%49.17%50.83%47.58%52.42%48.59%51.41%
Overall 41.60%58.40%49.59%50.41%46.77%53.23%48.19%51.81%
LegalComprehensive 43.60%50.80%49.60%50.40%44.80%55.20%48.00%52.00%
Diversity 41.60%51.20%44.00%56.00%36.80%63.20%42.80%57.20%
Empowerment 42.00%51.60%50.00% 50.00% 45.20%54.80%48.00%52.00%
Overall 42.40%51.60%50.00% 50.00% 44.80%55.20%47.60%52.40%
MixComprehensive 45.60%54.40%48.80%51.20%45.20%54.80%49.60%50.40%
Diversity 40.80%59.20%51.60% 48.40% 38.40%61.60%45.20%54.80%
Empowerment 45.60%54.40%47.60%52.40%42.40%57.60%49.20%50.80%
Overall 46.00%54.00%48.40%51.60%42.40%57.60%49.40%50.60%
CS Legal Agriculture Mix3540455055606570Winning Rate (%)Blocking Type
Semantic
Entity Type
Structural
CS Legal Agriculture Mix3540455055606570Winning Rate (%)Entity Embedding
Qwen3-Embedding-8B
CompGCN
RGCNComplEx
DistMult
TransE
CS Legal Agriculture Mix3540455055606570Winning Rate (%)Similarity Mode
Ego node
Neighbor Only
Ego + NeighborNeighbor subset Only
Ego + Neighbor subset
CS Legal Agriculture Mix3540455055606570Winning Rate (%)Merge Type
Direct Merging
Merging + Synonym Linking
Synonym Linking
Figure 3: Impact of different entity resolution components on Graph-based RAG performance.
8

Preprint.
relations as vectors in a complex number vector space to better handle asymmetric relations. This
demonstrates that traditional KG embeddings can be a viable alternative to LLM embeddings, es-
pecially in scenarios where computational resources are insufficient for LLMs or when we contain
complex relations in the datasets. (3) Without ego-based similarity, the performance of Graph-based
RAG degrades in most cases. Additionally, incorporating neighbor information as a complement
to ego node information improves performance in theLegalandMixdatasets. (4) Simple direct
merging often surpasses synonym linking. Although both methods aim to deal with the synonym
entities, synonym linking only adds synonym relations between merged entities and the canonical
entity. As a result, the KGs remain redundant, requiring more hops to retrieve relevant informa-
tion. In contrast, direct merging addresses this by consolidating entities with similar meanings into
a single entity, which is more efficient.
5.4 HYPERPARAMETERANALYSIS
0.2 0.4 0.6 0.8
Node Reduction Ratio40506070Winning Rate (%)
49.452.764.3
59.6
55.7
53.0
48.8
39.6CS
0.2 0.4 0.6 0.8
Node Reduction Ratio405060Winning Rate (%)
48.349.551.951.353.5
50.951.3
40.4Legal
0.2 0.4 0.6 0.8
Node Reduction Ratio506070Winning Rate (%)
57.164.9
58.157.6
54.5
46.346.949.9Agriculture
0.2 0.4 0.6 0.8
Node Reduction Ratio40506070Winning Rate (%)
55.253.752.255.560.7
57.056.6
41.7Mix
Figure 4: Influence of entity reduction ratio on Graph-based RAG performance.
We conduct experiments to investigate the robustness of the selection of the entity reduction ratio
on the effectiveness of denoising. As shown in Figure 4, the winning rate is equal or larger than
50% as long as reduction ratio is not too high. This means, as long as entities are not over-merged,
the denoising step is effective for Graph-based RAG. Notably, onMixandLegal, the performance
remains comparable to the original KG up to70%, which means even the reduction of70%entities
in KG does not cause negative effect compared to original KG. At such aggressive denoising setting,
not only near-duplicate or synonymous entities are merged, but entities with only marginal semantic
similarity and overlapping local neighborhoods can also be absorbed into a single canonical node,
effectively collapsing fine-grained clusters. The resulting KG becomes substantially more compact
while still keep, and sometimes even improve, Graph-based RAG performance. We attribute this to
the reduced redundancy, shorter multi-hop paths, and the concentration on fewer, more informative
nodes. This indicates that Graph-based RAG is robust to some over-merging cases so long as coarse-
grained semantics are preserved.
5.5 ABLATIONSTUDY
Agriculture CS Legal Mix4550556065Winning Rate (%)Full Method
w/o Entity
Resolutionw/o Triple
Reflection
Random
Merging
Figure 5: Ablation study on the performance of
the full denoising method against versions with-
out entity resolution, without triple reflection, and
with random entity merging.To evaluate the effectiveness of entity reso-
lution and triple reflection in DEG-RAG, we
conduct an ablation study in this subsection.
As shown in Figure 5, without entity reso-
lution or triple reflection, the performance of
Graph-based RAG significantly degrades in all
datasets. Moreover, we find that entity reso-
lution is more impactful than triple reflection,
indicating the necessity of entity resolution in
KGs. We also set up random merging as a ref-
erence method for comparison and the results
show worse performance than the above two
partial methods, which again shows the neces-
sity to handle the redundant entities smartly.
6 CONCLUSION ANDFUTUREWORKS
In this work, we investigated how denoising LLM-generated KGs benefits Graph-based RAG. We
introduced DEG-RAG, which combines entity resolution and triple reflection to remove redundant
9

Preprint.
entities and filter unreliable relations. Across four Graph-based RAG variants and four datasets,
DEG-RAGreduces around half the size of the entities and relations while preserving or improving
QA quality and lowering storage cost. Our component analysis shows that type-aware blocking is
consistently strong, classical KG embeddings such as ComplEx can rival LLM embeddings, ego in-
formation is essential and neighbor cues help in some settings, and direct merging generally outper-
forms synonym-only linking. Hyperparameter sweeps reveal wide operating regimes and sometimes
allow up to 70% entity reduction without hurting performance. Our methods focus on improving
the quality of KGs and can be used alongside advances in knowledge-graph-based LLM applications
(Choudhary & Reddy, 2023; Wang et al., 2025; Wang, 2025).
While effective, DEG-RAGhas limitations. Our study uses four QA datasets and non-large-scale
KGs. Triple reflection depends on LLM prompting and the LLM-as-judge setup, which can intro-
duce calibration bias. Gains are bounded by attribute richness. For example, graphs with only short
names without rich descriptions limit resolution quality. In future work, we will extend DEG-RAGto
more datasets and larger-scale KGs, generalize the denoising pipeline to other LLM-generated data
structures beyond KGs, and richer evaluations beyond LLM as judges.
ACKNOWLEDGMENT
Supported by the Overseas Open Funds of State Key Laboratory for Novel Software Technology of
China ( No.KFKT2025A06)
REPRODUCIBILITY STATEMENT
We have provided the codebase in supplementary material and all the results in this paper are repro-
ducible. The additional implementation details and experimental setups can be found in Section 5.1
and Appendix C.
ETHICS STATEMENT
All of the authors in this paper have read and followed the ethics code.
REFERENCES
Hiba Abu Ahmad and Hongzhi Wang. An effective weighted rule-based method for entity resolution.
Distributed and Parallel Databases, 36(3):593–612, 2018.
Mehdi Ali, Max Berrendorf, Charles Tapley Hoyt, Laurent Vermue, Sahand Sharifzadeh, V olker
Tresp, and Jens Lehmann. PyKEEN 1.0: A Python Library for Training and Evaluating Knowl-
edge Graph Embeddings.Journal of Machine Learning Research, 22(82):1–6, 2021. URL
http://jmlr.org/papers/v22/20-825.html.
Max Berrendorf, Evgeniy Faerman, Valentyn Melnychuk, V olker Tresp, and Thomas Seidl. Knowl-
edge graph entity alignment with graph convolutional networks: Lessons learned. InEuropean
Conference on Information Retrieval, pp. 3–11. Springer, 2020.
Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko.
Translating embeddings for modeling multi-relational data.Advances in neural information pro-
cessing systems, 26, 2013.
Nurendra Choudhary and Chandan K Reddy. Complex logical reasoning over knowledge graphs
using large language models.arXiv preprint arXiv:2305.01157, 2023.
Vassilis Christophides, Vasilis Efthymiou, Themis Palpanas, George Papadakis, and Kostas Ste-
fanidis. An overview of end-to-end entity resolution for big data.ACM Comput. Surv., 53(6),
December 2020. ISSN 0360-0300. doi: 10.1145/3418896. URLhttps://doi.org/10.
1145/3418896.
10

Preprint.
Gheorghe Comanici et al. Gemini 2.5: Pushing the frontier with advanced reasoning, mul-
timodality, long context, and next generation agentic capabilities. arXiv:2507.06261, 2025.
URLhttps://storage.googleapis.com/deepmind-media/gemini/gemini_
v2_5_report.pdf. Google DeepMind Technical Report.
Muhammad Ebraheem, Saravanan Thirumuruganathan, Shafiq Joty, Mourad Ouzzani, and Nan
Tang. Deeper–deep entity resolution.arXiv preprint arXiv:1710.00597, 2017.
Darren Edge, Ha Trinh, N Cheng, J Bradley, A Chao, A Mody, S Truitt, and J Larson. From local
to global: A graph rag approach to query-focused summarization. arxiv 2024.arXiv preprint
arXiv:2404.16130, 2024.
Vasilis Efthymiou, George Papadakis, Kostas Stefanidis, and Vassilis Christophides. Minoaner:
Schema-agnostic, non-iterative, massively parallel resolution of web entities.arXiv preprint
arXiv:1905.06170, 2019.
Yiyang Feng, Yichen Wang, Shaobo Cui, Boi Faltings, Mina Lee, and Jiawei Zhou. Unraveling
misinformation propagation in llm reasoning.arXiv preprint arXiv:2505.18555, 2025.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-
augmented generation.(2024).arXiv preprint arXiv:2410.05779, 2024.
Ridong Han, Chaohao Yang, Tao Peng, Prayag Tiwari, Xiang Wan, Lu Liu, and Benyou Wang.
An empirical study on information extraction using large language models.arXiv preprint
arXiv:2305.14450, 2023.
Yubin Hong, Chaofan Li, Jingyi Zhang, and Yingxia Shao. Fg-rag: Enhancing query-focused sum-
marization with context-aware fine-grained graph rag.arXiv preprint arXiv:2504.07103, 2025.
Haoyu Huang, Chong Chen, Zeang Sheng, Yang Li, and Wentao Zhang. Can llms be good graph
judge for knowledge graph construction?arXiv preprint arXiv:2411.17388, 2024a.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong
Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language
models: Principles, taxonomy, challenges, and open questions.ACM Transactions on Information
Systems, 43(2):1–55, 2025.
Yue Huang, Lichao Sun, Haoran Wang, Siyuan Wu, Qihui Zhang, Yuan Li, Chujie Gao, Yixin
Huang, Wenhan Lyu, Yixuan Zhang, et al. Trustllm: Trustworthiness in large language models.
arXiv preprint arXiv:2401.05561, 2024b.
Ziyan Jiang, Xueguang Ma, and Wenhu Chen. Longrag: Enhancing retrieval-augmented generation
with long-context llms.arXiv preprint arXiv:2406.15319, 2024.
Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobi-
ologically inspired long-term memory for large language models.Advances in Neural Information
Processing Systems, 37:59532–59569, 2024.
Hangzhan Jin, Sitao Luan, Sicheng Lyu, Guillaume Rabusseau, Reihaneh Rabbany, Doina Pre-
cup, and Mohammad Hamdaqa. Rl fine-tuning heals ood forgetting in sft.arXiv preprint
arXiv:2509.12235, 2025.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. InEMNLP
(1), pp. 6769–6781, 2020.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention. InProceedings of the ACM SIGOPS 29th Symposium on Operating
Systems Principles, 2023.
Yassir Lairgi, Ludovic Moncla, R ´emy Cazabet, Khalid Benabdeslem, and Pierre Cl ´eau. itext2kg:
Incremental knowledge graphs construction using large language models. InInternational Con-
ference on Web Information Systems Engineering, pp. 214–229. Springer, 2024.
11

Preprint.
Heeyoung Lee, Angel Chang, Yves Peirsman, Nathanael Chambers, Mihai Surdeanu, and Dan
Jurafsky. Deterministic coreference resolution based on entity-centric, precision-ranked rules.
Computational Linguistics, 39(4):885–916, December 2013. doi: 10.1162/COLI a00152. URL
https://aclanthology.org/J13-4004/.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, and Wang-Chiew Tan. Deep entity matching
with pre-trained language models.arXiv preprint arXiv:2004.00584, 2020.
Lei Liang, Mengshu Sun, Zhengke Gui, Zhongshu Zhu, Zhouyu Jiang, Ling Zhong, Peilong Zhao,
Zhongpu Bo, Jin Yang, et al. Kag: Boosting llms in professional domains via knowledge aug-
mented generation.arXiv preprint arXiv:2409.13731, 2024.
Jerry Liu. LlamaIndex, 11 2022. URLhttps://github.com/jerryjliu/llama_index.
Jinghui Lu, Yanjie Wang, Ziwei Yang, Xuejing Liu, Brian Mac Namee, and Can Huang. Padellm-
ner: parallel decoding in large language models for named entity recognition.Advances in Neural
Information Processing Systems, 37:117853–117880, 2024.
Wensheng Lu, Keyu Chen, Ruizhi Qiao, and Xing Sun. Hichunk: Evaluating and enhancing
retrieval-augmented generation with hierarchical chunking.arXiv preprint arXiv:2509.11552,
2025.
Sitao Luan, Chenqing Hua, Qincheng Lu, Jiaqi Zhu, Mingde Zhao, Shuyuan Zhang, Xiao-Wen
Chang, and Doina Precup. Revisiting heterophily for graph neural networks.Advances in neural
information processing systems, 35:1362–1375, 2022.
Sitao Luan, Chenqing Hua, Qincheng Lu, Liheng Ma, Lirong Wu, Xinyu Wang, Minkai Xu,
Xiao-Wen Chang, Doina Precup, Rex Ying, et al. The heterophilic graph learning hand-
book: Benchmarks, models, theoretical analysis, applications and challenges.arXiv preprint
arXiv:2407.09618, 2024.
Birger Mo ¨ell and Fredrik Sand Aronsson. Harm reduction strategies for thoughtful use of large
language models in the medical domain: perspectives for patients and clinicians.Journal of
Medical Internet Research, 27:e75849, 2025.
Daniel Obraczka, Jonathan Schuchart, and Erhard Rahm. Eager: embedding-assisted entity resolu-
tion for knowledge graphs.arXiv preprint arXiv:2101.06126, 2021.
OpenAI. Gpt-4o mini: Advancing cost-efficient intelligence.https://openai.com/
index/gpt-4o-mini-advancing-cost-efficient-intelligence/, July 2024.
Accessed: 2025-09-24.
Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. The pagerank citation ranking:
Bringing order to the web. Technical Report 1999-66, Stanford InfoLab, 1998. URLhttp:
//ilpubs.stanford.edu:8090/422/.
George Papadakis, Dimitrios Skoutas, Emmanouil Thanos, and Themis Palpanas. A survey of block-
ing and filtering techniques for entity resolution.arXiv preprint arXiv:1905.06167, 2019.
George Papadakis, Vasilis Efthymiou, Emmanouil Thanos, Oktie Hassanzadeh, and Peter Christen.
An analysis of one-to-one matching algorithms for entity resolution.The VLDB Journal, 32(6):
1369–1400, April 2023. ISSN 1066-8888. doi: 10.1007/s00778-023-00791-3. URLhttps:
//doi.org/10.1007/s00778-023-00791-3.
Ralph Peeters, Aaron Steiner, and Christian Bizer. Entity matching using large language models.
arXiv preprint arXiv:2310.11244, 2023.
12

Preprint.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and
Siliang Tang. Graph retrieval-augmented generation: A survey.arXiv preprint arXiv:2408.08921,
2024.
Jay Pujara and Lise Getoor. Generic statistical relational entity resolution in knowledge graphs.
arXiv preprint arXiv:1607.00992, 2016.
Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and Tiejun
Huang. Memorag: Boosting long context processing with global memory-enhanced retrieval
augmentation. InProceedings of the ACM Web Conference 2025 (TheWebConf 2025), Sydney,
Australia, 2025. ACM. URLhttps://arxiv.org/abs/2409.05591.
Toni C. M. Rietveld, Roeland van Hout, and Mirjam Ernestus. Pitfalls in corpus research.Comput.
Humanit., 38(4):343–362, 2004. doi: 10.1007/S10579-004-1919-1. URLhttps://doi.
org/10.1007/s10579-004-1919-1.
Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan Titov, and Max
Welling. Modeling relational data with graph convolutional networks. InEuropean semantic web
conference, pp. 593–607. Springer, 2018.
Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana, and
Suranga Nanayakkara. Improving the domain adaptation of retrieval augmented generation (rag)
models for open domain question answering.Transactions of the Association for Computational
Linguistics, 11:1–17, 2023.
Aaron Steiner, Ralph Peeters, and Christian Bizer. Fine-tuning large language models for en-
tity matching. In2025 IEEE 41st International Conference on Data Engineering Workshops
(ICDEW), pp. 9–17. IEEE, 2025.
Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. Dragin: dynamic retrieval
augmented generation based on the information needs of large language models.arXiv preprint
arXiv:2403.10081, 2024.
Qwen Team. Qwen3 technical report, 2025. URLhttps://arxiv.org/abs/2505.09388.
Th´eo Trouillon, Johannes Welbl, Sebastian Riedel, ´Eric Gaussier, and Guillaume Bouchard. Com-
plex embeddings for simple link prediction. InInternational conference on machine learning, pp.
2071–2080. PMLR, 2016.
Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, and Partha Talukdar. Composition-based multi-
relational graph convolutional networks.arXiv preprint arXiv:1911.03082, 2019.
Nan Wang, Yongqi Fan, ZongYu Wang, Xuezhi Cao, Xinyan He, Haiyun Jiang, Tong Ruan, Jing-
ping Liu, et al. Kg-o1: Enhancing multi-hop question answering in large language models via
knowledge graph integration.arXiv preprint arXiv:2508.15790, 2025.
Shaofei Wang. Enhancing in-context learning of large language models for knowledge graph rea-
soning via rule-and-reinforce selected triples.Applied Sciences, 15(3):1088, 2025.
Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. Aug-
menting language models with long-term memory.Advances in Neural Information Processing
Systems, 36:74530–74543, 2023.
Bingcong Xue and Lei Zou. Knowledge graph quality management: A comprehensive survey.IEEE
Transactions on Knowledge and Data Engineering, 35(5):4969–4988, 2022.
Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and
relations for learning and inference in knowledge bases.arXiv preprint arXiv:1412.6575, 2014.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering.arXiv preprint arXiv:1809.09600, 2018.
13

Preprint.
Minghe Yu, Guoliang Li, Dong Deng, and Jianhua Feng. String similarity search and join: a survey.
Frontiers Comput. Sci., 10(3):399–417, 2016. doi: 10.1007/S11704-015-5900-5. URLhttps:
//doi.org/10.1007/s11704-015-5900-5.
Yihong Yuan and Martin Raubal. Extracting dynamic urban mobility patterns from mobile phone
data. InInternational conference on geographic information science, pp. 354–367. Springer,
2012.
Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and
Joseph E Gonzalez. Raft: Adapting language model to domain specific rag.arXiv preprint
arXiv:2403.10131, 2024.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie,
An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advanc-
ing text embedding and reranking through foundation models.arXiv preprint arXiv:2506.05176,
2025.
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min,
Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models.arXiv
preprint arXiv:2303.18223, 1(2), 2023.
Yilun Zheng, Sitao Luan, and Lihui Chen. What is missing for graph homophily? disentangling
graph homophily for graph neural networks.Advances in Neural Information Processing Systems,
37:68406–68452, 2024.
Yilun Zheng, Xiang Li, Sitao Luan, Xiaojiang Peng, and Lihui Chen. Let your features tell the
differences: Understanding graph convolution by feature splitting. InThe Thirteenth International
Conference on Learning Representations, 2025.
Zijie Zhong, Hanwen Liu, Xiaoya Cui, Xiaofan Zhang, and Zengchang Qin. Mix-of-granularity:
Optimize the chunking granularity for retrieval-augmented generation. In Owen Rambow, Leo
Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert
(eds.),Proceedings of the 31st International Conference on Computational Linguistics, pp. 5756–
5774, Abu Dhabi, UAE, January 2025. Association for Computational Linguistics.
Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, Runyuan He, Yongwei Zhang,
Sicong Liang, Xilin Liu, Yuchi Ma, et al. In-depth analysis of graph-based rag in a unified
framework.arXiv preprint arXiv:2503.04338, 2025.
14

Preprint.
THEUSE OFLARGELANGUAGEMODELS
In this work, we employed LLMs as auxiliary tools to support the preparation of the manuscript.
Specifically, LLMs were used in two ways: (i) to polish the writing style of the paper by refining
grammar, clarity, and readability without altering the technical content, and (ii) to assist in identify-
ing relevant related work by suggesting potential references. Note that LLMs were not involved in
designing experiments, analyzing results, or drawing conclusions; these aspects of the study were
carried out independently by the authors.
A DATASTATISTICS
Table 2: Statistics of datasets and knowledge graphs across four domains.
Category Agriculture CS Legal Mix
# Token 1,949,526 2,047,866 4,872,343 611,161
# Document 12 10 94 61
# Question 125 125 125 125
# EntityLightRAG 21,131 16,434 16,502 8,942
HippoRAG 42,444 25,495 34,342 24,055
LGraphRAG 21,761 15,257 16,761 10,240
GGraphRAG 21,227 15,600 16,111 10,399
# RelationLightRAG 23,102 20,642 33,625 7,458
HippoRAG 41,636 25,170 51,031 16,370
LGraphRAG 25,834 19,980 36,742 8,513
GGraphRAG 21,408 19,412 36,507 9,943
Ave. Entity
DescriptionLightRAG 40.47 42.12 63.64 32.61
HippoRAG – – – –
LGraphRAG 40.23 40.21 62.11 31.88
GGraphRAG 38.74 39.83 63.76 33.66
As shown in Table A, we report the numbers of tokens, documents, and questions for the four
datasets used in this paper. We also present the counts of entities and relations, as well as the
average length of entity descriptions (in tokens) in the LLM-generated knowledge graphs extracted
by LightRAG (Guo et al., 2024), HippoRAG (Jimenez Gutierrez et al., 2024), LGraphRAG (Edge
et al., 2024), and GGraphRAG (Edge et al., 2024). Note that the knowledge graphs generated by
HippoRAG do not contain entity descriptions.
B ADDITIONALEXPERIMENTALRESULTS
B.1 IMPACT OFDIFFERENTLLMS INRAG
To show how different LLMs backbones influences the performance of DEG-RAGshown in Table
5.1, apart from Qwen3-235B-A22B (Team, 2025), we further conduct experiments using GPT-4o-
mini (OpenAI, 2024) and Gemini-2.5-flash (Comanici et al., 2025) on four datasets on LightRAG
(Guo et al., 2024). As shown in Table B.1, under the entity reduction of40%and triple reflec-
tion threshold of0.2, the winning rate of using GPT-4o-mini or Gemini-2.5-flash is comparable as
Qwen3-235-A22B, indicating the generality of DEG-RAGacross different types of LLMs.
B.2 COMPARISON OFTOKENCONSUMPTION
We further compare the costs of DEG-RAGunder different entity reduction ratios. Table B.2 shows
the statistics of token consumption after applying DEG-RAGin LightRAG as shown in Table 5.1.
First, we can see that there is no significant differences of token consumption in prompt and com-
pletion for LightRAG on the original knowledge graph and knowledge graphs with DEG-RAG, in-
dicating the performance gain is not caused by additional information. Second, we notice that the
15

Preprint.
Table 3: Performance comparison of models on original and cleaned knowledge graphs across four
datasets. The evaluation is based on winning rates by comparing responses generated from original
versus cleaned knowledge graphs.
Dataset DimensionQwen3-235B-A22B GPT-4o-mini Gemini-2.5-flash
Orig. Clean Orig. Clean Orig. Clean
AgricultureComprehensive 43.60%56.40%45.34%54.66%46.00%54.00%
Diversity 41.60%58.40%29.27%70.73%46.00%54.00%
Empowerment 42.00%58.00%31.71%68.29%46.80%53.20%
Overall 42.40%57.60%33.74%66.26%46.80%53.20%
CSComprehensive 39.20%60.80%42.32%57.68%44.40%55.60%
Diversity 40.00%60.00%36.51%63.49%43.20%55.60%
Empowerment 40.80%59.20%41.91%58.09%43.20%56.80%
Overall 41.60%58.40%41.91%58.09%44.00%56.00%
LegalComprehensive 43.60%56.40%46.40%53.60%42.00%58.00%
Diversity 41.60%58.40%45.20%54.80%42.40%57.60%
Empowerment 42.00%58.00%46.80%53.20%40.80%59.20%
Overall 42.40%57.60%47.60%52.40%41.20%58.80%
MixComprehensive 45.60%54.40%47.18%52.82%42.40%57.60%
Diversity 40.80%59.20%43.95%56.05%40.00%60.00%
Empowerment 45.60%54.40%45.16%54.84%42.40%57.60%
Overall 46.00%54.00%45.56%54.44%42.00%58.00%
Table 4: Token consumption statistics under different entity reduction ratios across four datasets.
Dataset Type Original 20% 40% 60% 80%
MixPrompt 1,040,189 1,185,787 1,267,955 1,149,659 1,133,338
Completion 86,171 85,738 85,454 85,334 86,051
Total Token 1,126,360 1,271,525 1,353,409 1,234,993 1,219,389
CSPrompt 1,084,623 1,118,326 1,106,513 906,618 779,191
Completion 89,056 90,658 89,394 88,844 89,252
Total Token 1,173,679 1,208,984 1,195,907 995,462 868,443
AgriculturePrompt 1,273,710 1,537,191 1,296,947 1,278,717 911,124
Completion 82,351 82,677 82,978 79,724 79,683
Total Token 1,356,061 1,619,868 1,379,925 1,358,441 990,807
LegalPrompt 1,755,056 1,749,183 1,721,740 1,528,838 1,658,700
Completion 84,124 84,771 84,178 83,707 85,468
Total Token 1,839,180 1,833,954 1,805,918 1,612,545 1,744,168
input token increases with node reduction of20%or40%, then decreases on60%and80%. We
explain this as, in lower reduction ratio, few entities are merged, which slightly increases the input
prompt, while in high reduction ratio, more and more entities are merged together, after the sum-
maziation of entitiy description, the total retrieved entites and relations become fewer, leads to fewer
input token.
B.3 CASESTUDY
To illustrate the qualitative impact of denoising, we conduct a case study on entity resolution using
the CS dataset. Figure 6 shows a subgraph of the knowledge graph before and after denoising.
Red nodes indicate redundant entities that have been merged into their canonical forms, while blue
nodes represent entities that remain unchanged. Dashed red lines indicate the direction of merging
from one entity to another, green lines denote newly added relations, brown dashed lines represent
removed relations, and black lines correspond to relations that are retained.
The entity merging process is generally reasonable. For example, variations such asARIME
methodologyare merged intoARIMA model, andLinear Regressionintolinear
16

Preprint.
ARIMA
ModelARIMA
methodology
Forecast
PackageTime
Series
Analysis
Expected
Value
E(X)
Weak Law
of Large
NumbersProbability
of Distribution
P(X)Probability
of Distribution
Sample
MeansDataset
{X}covariance
matrix
Covariance
Matrix
DatasetSample
MeanCorrelation
Coefficient
CorrelationCorrelation
Coefficient
(r)
Bootstrap
Replicatessales
forecastPSO
Algorithm
meanlinear
regressionLinear
RegressionLinear
Models
Decision
TreeClustering
models
Confidence
Interval
Bootstrap
MethodR 
programming
Environment
Normalized
Coordinates
DatasetsNegative
Correlation
Probability
Theory
Chebyshev's
InequalityK-means
AlgorithmUnsupervised
Learning
Naive
Bayes
Model
Support
Vector 
Machine
ModelL1
Regulari_
zationL2
Regulari-
zation
Figure 6: Case Study of Knowledge Graph Denoising on the CS Dataset. The figure illustrates a
subgraph before and after applying our denoising method. Redundant entities are denoted in red and
merging process is shown in arrows.
regression. We also observe merges driven by semantic similarity, such asK-means
Algorithmbeing merged intoClustering models, andNaive Bayes Modeland
Support Vector Machine Modelbeing merged intoDecision Tree. Overall, the de-
noised knowledge graph is more concise and efficient, thereby improving the performance of graph-
based RAG.
We also examine the cases of triple reflection. As in Table 5, we listed some triples withδ TR≤0.2.
C IMPLEMENTATIONDETAILS
C.1 GRAPH-BASEDRAG
For all the Graph-based RAG methods, we set token-based chunking across all methods, with seg-
ment length of approximately 1,200 tokens and an overlap of 100 tokens, using a standard tokenizer
to balance context preservation and indexing granularity. We set the retriever to return the top 5
candidates. When personalized PageRank is used, we set entity-aware priors with light damping to
encourage focus on salient nodes. All methods answer questions directly rather than only returning
supporting context. We set the overall candidate pool to 20. We set token budgets consistently across
methods: the naive assembly budget to 12,000 tokens, the local assembly budget to 4,000 tokens,
and the entity and relation evidence budgets to 2,000 tokens each. When iterative reasoning over
retrieved evidence is enabled, we cap the refinement steps at 2.
LightRAG (Guo et al., 2024) maintains both entity and relation indices and builds a relation-centric
knowledge graph enriched with edge keywords. We enable entity descriptions, entity types, edge
descriptions, and edge names to maximize semantic coverage. We set the usable context window
to 32,768 tokens. For retrieval, we set nearest-neighbor search and enable entity-similarity–aware
propagation with the top 5 results. Querying is hybrid: we enable both local and global graph search.
We set the global community cap to 512 without a minimum rating, the global community report
budget to 16,384 tokens, and the global context budget to 4,000 tokens. Locally, we set the context
budget to 4,800 tokens and the community report budget to 3,200 tokens. We allow keyword cues
when composing the final context.
17

Preprint.
Table 5: Case study of triple reflection
Relation Source Score Target Analysis
Transenterix Inc.
owns Safestitch
LLCTransenterix Inc.
owns Safestitch
LLC, indicating a
parent subsidiary
relationship0.1 Safestitch LLC TransEnterix does not own
SafeStitch
Turtle is one of
the entities classi-
fied as a borrowerTurtle is one of
the entities classi-
fied as a borrower
in the financial
agreement0.1 Borrowers Turtles are not entities that en-
gage in borrowing
Michael Scott is
involved in the
SEC lawsuitMichael Scott is
involved in the
SEC lawsuit as
a defendant ac-
cused of securi-
ties violations0.1 SEC lawsuit Michael Scott is a fictional char-
acter from the television show
’The Office’ and not a real per-
son involved in any legal mat-
ters
Title policy for
PabstTitle policy is re-
quired to obtain
a title policy to
ensure the legiti-
macy of the asset
ownership during
the acquisition0.1 Pabst A title policy is a type of insur-
ance related to real estate trans-
actions, while ’pabst’ appears to
refer to a brand
Shareholder’s eq-
uity reflects net
worth of dealersShareholder’s
equity is a key
financial metric
that reflects the
net worth of
dealers after
liabilities are
deducted0.2 Dealers Shareholder’s equity is a finan-
cial metric relevant to com-
panies and their owners, not
specifically to dealers
Kristen M Jenner
and Kylie K Jen-
ner are key exec-
utivesKylie K Jenner
and Kristen
M Jenner are
both identified
as key execu-
tives, indicating
a professional
relationship in a
business context0.2 Kylie K Jenner Kristen M Jenner is not a recog-
nized executive in the same con-
text as Kylie K Jenner
HippoRAG (Jimenez Gutierrez et al., 2024) focuses on an entity–relation graph with entity-
link–aware chunking and enables graph augmentation while keeping metadata conservative: we
disable entity and edge descriptions, and we retain edge names. We set retrieval to personalized
PageRank over the entity–relation graph without an entity-similarity term in propagation, and we
set the top-kto 5. Querying follows a hybrid strategy while we disable explicit propagation-based
augmentation in the final context assembly. We keep the same token budgets as in the common
configuration, and we cap iterative reasoning at 2 steps.
LGraphRAG (Edge et al., 2024) uses a relation-centric knowledge graph with a forced construction
setting. We enable entity and edge descriptions and edge names, and we disable entity types. We
apply community-aware clustering using the Leiden algorithm; we set the maximum community
size to 10 and use concise community summaries. We set retrieval to nearest-neighbor search with
an additional local neighborhood expansion, and we enable propagation-based augmentation while
disabling global community selection. We set the local context budget to 4,800 tokens and the local
18

Preprint.
community report budget to 3,200 tokens, and we keep the same overall budgets and refinement
limits as in the common setup.
GGraphRAG (Edge et al., 2024) adopts the same relation-centric graph construction and
community-aware clustering as LGraphRAG. We set retrieval to nearest-neighbor search without
local expansion, and we enable both local and global querying. We set the global community cap to
512, the global community report budget to 16,384 tokens, and the global context budget to 4,000
tokens, while keeping the local budgets aligned with the common configuration. Other token allo-
cations and refinement limits follow the common setup.
C.2 REDUCTIONRATIO
We further report the number and proportion of removed entities and relations in Table 5.1. As
shown in Table 6, across the four datasets, the entity reduction ratio is approximately40%. The
relation reduction ratio ranges from30%to60%, reflecting both the removal of relations during
triple reflection and the disappearance of relations associated with merged entities.
Table 6: Statistics of original and cleaned knowledge graphs across four datasets and four Graph-
based RAG models.
Dataset DimensionLightRAG HippoRAG LGraphRAG GGraphRAG
Orig. Clean Reduction Orig. Clean Reduction Orig. Clean Reduction Orig. Clean Reduction
Agriculture# Entity 21131 12679 40.00% 42444 25466 40.00% 21761 13057 40.00% 21227 12736 40.00%
# Relation 23102 15548 32.70% 41636 20321 51.19% 25834 16503 36.12% 21408 11258 47.41%
CS# Entity 16434 9861 40.00% 25495 15297 40.00% 15257 9154 40.00% 15600 9360 40.00%
# Relation 20642 12164 41.07% 25170 13801 45.17% 19980 13756 33.15% 19412 13742 29.21%
Legal# Entity 16502 9902 40.00% 34342 20606 40.00% 16761 10057 40.00% 16111 9667 40.00%
# Relation 33625 21261 36.77% 51031 35920 29.61% 36742 22987 37.44% 36507 14025 61.58%
Mix# Entity 8942 5366 40.00% 24055 14433 40.00% 10240 6144 40.00% 10399 6240 40.00%
# Relation 7458 5164 30.76% 16370 6896 57.87% 8513 6288 26.14% 9943 6713 32.49%
C.3 PROMPTS INENTITYRESOLUTION
To avoid the exceeding length of descriptions of merged knoweldge graphs, we summarize the
descriptions if the number of token exceed 4,000. We provide the summarization prompt of entity
and relation as follows
Entity description summarization prompt
You are a helpful assistant. Please summarize the following list of descriptions for the
entity{entity name}into a single, coherent paragraph. Combine the key information
and remove redundant details.
Descriptions to summarize:
{description list}
Concise Summary:
Relation description summarization prompt
You are a helpful assistant. Please summarize the following list of descriptions for the
relationship{item name}into a single, coherent paragraph. Combine the key information
and remove redundant details.
Descriptions to summarize:
{description list}
Concise Summary:
19

Preprint.
C.4 PROMPTS INTRIPLEREFLECTION
We perform triple reflection on knowledge graph triples (edges) using LLMs to assess their reason-
ableness before downstream use. For each triple, an LLM returns a numerical quality score and a
short analysis; results are written as JSONL for subsequent aggregation and filtering.
System prompt
You are a knowledge graph expert who evaluates whether the knowledge graph triplet be-
longs to commonsense knowledge.
User prompt
Evaluate the reasonableness of the knowledge graph triplet with precision:
Source:<source>
Destination:<destination>
Relationship:<relationship>
Analysis requirements
•Semantic accuracy: Does the relationship accurately describe the connection? Consider
domain knowledge and factual correctness.
•Relevance: Is the connection meaningful and significant, not trivial or coincidental?
•Specificity: Is the relationship clear and specific rather than vague or overly general?
•Logical coherence: Does the triple follow expected semantic and syntactic patterns for
KGs?
•Entity type compatibility: Is the relationship sensible given the entity types involved?
Scoring guidelines
•0.0–0.3: Invalid or highly questionable (factually wrong, illogical, meaningless)
•0.4–0.6: Partially valid but problematic (some relevance yet vague/imprecise/minor inac-
curacies)
•0.7–0.8: Mostly valid (accurate but could be more specific or informative)
•0.9–1.0: Fully valid (accurate, specific, informative, and logically sound)
Optimization notes
• Focus on direct evaluation without unnecessary elaboration.
• Use domain-specific reasoning where applicable.
Output format (return a valid JSON object):
{
"analysis": "concise analysis",
"score": 0.5
}
The score should be a float between 0.0–1.0 with two-decimal precision.
C.5 EVALUATION
We assess the responses of DEG-RAGusing an LLM judge in a pairwise-comparison setup. For
each question the judge receives the question and two candidate answers from original knowledge
graphs or denoised knowledge graphs by DEG-RAG, and decides which answer is better and why.
To mitigate position bias we run two passes per question. Pass A uses (Answer 1, Answer 2) and
Pass B swaps the order. Aggregated wins for a method on a criterion are computed by summing
Answer 1 wins in Pass A and Answer 2 wins in Pass B. Ties are recorded when the judge issues a
tie token. The judge receives the following prompts verbatim.
20

Preprint.
System prompt
You are an expert tasked with evaluating two answers to the same question based on three
criteria:Comprehensiveness,Diversity, andEmpowerment.
User prompt
You will evaluate two answers to the same question using the three criteria below:
•Comprehensiveness: How much detail does the answer provide to cover all aspects and
details of the question?
•Diversity: How varied and rich is the answer in presenting different perspectives and
insights?
•Empowerment: How well does the answer help the reader understand the topic and make
informed judgments?
For each criterion, choose the better answer (Answer 1orAnswer 2) and explain why. Then
select an overall winner based on these three categories.
Here is the question:{query}
Here are the two answers:
Answer 1:{answer1}
Answer 2:{answer2}
Evaluate both answers using the three criteria above and provide detailed explanations for
each criterion.
Output your evaluation in the following JSON format:
{
"Comprehensiveness": {
"Winner": "[Answer 1 or Answer 2]",
"Explanation": "[Provide explanation here]"
},
"Diversity": {
"Winner": "[Answer 1 or Answer 2]",
"Explanation": "[Provide explanation here]"
},
"Empowerment": {
"Winner": "[Answer 1 or Answer 2]",
"Explanation": "[Provide explanation here]"
},
"Overall Winner": {
"Winner": "[Answer 1 or Answer 2]",
"Explanation": "[Summarize why this answer is the
overall winner based on the three criteria]"
}
}
D PROOF OFPROPOSITION1
Proposition 1.Given a graph-based RAG and a vanilla RAG system that share the same augmen-
tation and generation processes, the absence of entity resolution causes the graph-based RAG to
degrade into vanilla RAG.
Proof.We assume that: (1) both systems use identical augmentation and generation processes except
for the knowledge representation, (2) vanilla RAG retrieves chunks based on relevance scoring, and
(3) graph-based RAG retrieves subgraphs or triples based on query-entity matching. This is not a
formal proof but rather an intuitive argument.
Given document chunksC={c 1, . . . , c M}, a Graph-based RAG system constructs a knowledge
graphG∗= (E∗,R∗,T∗,A∗)through named entity recognition followed by deduplication. The
21

Preprint.
responseYis generated for queryQas:
Y=M ◦Aug
Q,Ret(Q,G∗)
.(12)
Without entity resolution, the deduplication function becomes the identity mappingϕ(e) =efor all
e∈ E raw. This means:
E∗={ϕ(e)|e∈ E raw}=E raw (13)
T∗=T raw (14)
A∗(e) =A raw(e)∀e∈ E∗(15)
Since each triple(e 1, r, e 2)∈ T raworiginates from a single chunkc m, and no entity merging occurs,
entities from different chunks remain disconnected even if they represent the same real-world con-
cept. Formally, letE m={e 1, e2|(e1, r, e 2)∈ T m}be entities extracted from chunkc m. Without
entity resolution, there are no edges connecting entities from different chunks:
∀i̸=j:N(e i)∩ E j=∅wheree i∈ Ei (16)
This results inMdisconnected subgraphsG∗
1,G∗
2, . . . ,G∗
M, where eachG∗
m= (E m,Rm,Tm,Am)
corresponds to chunkc m.
For any queryQ, the graph retrieval function Ret(Q,G∗)can only retrieve from individual discon-
nected components. Since each componentG∗
mcontains only local information from chunkc m,
the retrieved content consists of triplesT mthat represent structured partitions of the original chunk
content. The graph-based retrieval without entity resolution becomes:
Ret(Q,G∗) =[
m:rel(Q,G∗m)>τTm (17)
where rel(Q,G∗
m)measures relevance between query and local subgraph, andτis a threshold.
Note that each original chunkc mcan be decomposed as:
cm=Tm∪unextracted text (18)
whereT mrepresents the structured information extracted fromc m. SinceT m⊂cm, the retrieved
triples are essentially parts of the original chunks. With no cross-chunk connections, this retrieval
process can be considered as a vanilla RAG system:
Ret vanilla(Q,{T m}) ={T m|rel(Q,T m)> τ′}(19)
for appropriately chosen thresholdsτandτ′.
Since the augmentation and generation processes are identical by assumption, and the retrieved
content has the same information coverage (parts of chunks vs. disconnected subgraphs), we have:
Ygraph=M ◦Aug[Q,Ret(Q,G∗)]≡ M ◦Aug[Q,Ret vanilla(Q,{T m})] =Y vanilla (20)
Therefore, without entity resolution, graph-based RAG degrades to vanilla RAG.
22