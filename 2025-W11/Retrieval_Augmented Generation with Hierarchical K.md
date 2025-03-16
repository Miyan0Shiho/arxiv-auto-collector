# Retrieval-Augmented Generation with Hierarchical Knowledge

**Authors**: Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu Pan, Yongqiang Chen, Kaili Ma, Hongzhi Chen, James Cheng

**Published**: 2025-03-13 08:22:31

**PDF URL**: [http://arxiv.org/pdf/2503.10150v1](http://arxiv.org/pdf/2503.10150v1)

## Abstract
Graph-based Retrieval-Augmented Generation (RAG) methods have significantly
enhanced the performance of large language models (LLMs) in domain-specific
tasks. However, existing RAG methods do not adequately utilize the naturally
inherent hierarchical knowledge in human cognition, which limits the
capabilities of RAG systems. In this paper, we introduce a new RAG approach,
called HiRAG, which utilizes hierarchical knowledge to enhance the semantic
understanding and structure capturing capabilities of RAG systems in the
indexing and retrieval processes. Our extensive experiments demonstrate that
HiRAG achieves significant performance improvements over the state-of-the-art
baseline methods. The code of our proposed method is available at
\href{https://github.com/hhy-huang/HiRAG}{https://github.com/hhy-huang/HiRAG}.

## Full Text


<!-- PDF content starts -->

Retrieval-Augmented Generation with Hierarchical Knowledge
Haoyu Huang1,2*, Yongfeng Huang2*, Junjie Yang2, Zhenyu Pan1,2, Yongqiang Chen1
Kaili Ma1, Hongzhi Chen1, James Cheng2
1KASMA.ai
2Department of Computer Science and Engineering, The Chinese University of Hong Kong
{haoyuhuang,zhenyupan,yqchen,klma,chenhongzhi}@kasma.ai
{haoyuhuang,zhenyupan,1155215805}@link.cuhk.edu.hk
{yfhuang22,jcheng}@cse.cuhk.edu.hk
Abstract
Graph-based Retrieval-Augmented Generation
(RAG) methods have significantly enhanced the
performance of large language models (LLMs)
in domain-specific tasks. However, existing
RAG methods do not adequately utilize the
naturally inherent hierarchical knowledge in
human cognition, which limits the capabilities
of RAG systems. In this paper, we introduce
a new RAG approach, called HiRAG, which
utilizes hierarchical knowledge to enhance the
semantic understanding and structure captur-
ing capabilities of RAG systems in the index-
ing and retrieval processes. Our extensive ex-
periments demonstrate that HiRAG achieves
significant performance improvements over
the state-of-the-art baseline methods. The
code of our proposed method is available at
https://github.com/hhy-huang/HiRAG.
1 Introduction
Retrieval Augmented Generation (RAG) (Gao
et al., 2023) (Lewis et al., 2020) (Fan et al., 2024)
has been introduced to enhance the capabilities of
LLMs in domain-specific or knowledge-intensive
tasks. Naive RAG methods retrieve text chunks that
are relevant to a query, which serve as references
for LLMs to generate responses, thus helping ad-
dress the problem of "Hallucination" (Zhang et al.,
2023) (Tang and Yang, 2024). However, naive
RAG methods usually overlook the relationships
among entities in the retrieved text chunks. To ad-
dress this issue, RAG systems with graph structures
were proposed (Edge et al., 2024) (Liang et al.,
2024) (Zhang et al., 2025) (Peng et al., 2024a),
which construct knowledge graphs (KGs) to model
relationships between entities in the input docu-
ments. Although existing RAG systems integrat-
ing graph structures have demonstrated outstand-
ing performance on various tasks, they still have
*Equal contribution. This research was conducted at
kasma.ai.some serious limitations. GraphRAG (Edge et al.,
2024) introduces communities in indexing using
the Leiden algorithm (Traag et al., 2019), but the
communities only capture the structural proxim-
ity of the entities in the KG. KAG (Liang et al.,
2024) indexes with a hierarchical representation of
information and knowledge, but their hierarchical
structure relies too much on manual annotation and
requires a lot of human domain knowledge, which
renders their method not scalable to general tasks.
LightRAG (Guo et al., 2024) utilizes a dual-level
retrieval approach to obtain local and global knowl-
edge as the contexts for a query, but it ignores the
knowledge gap between local and global knowl-
edge, that is, local knowledge represented by the
retrieved individual entities (i.e., entity-specific de-
tails) may not be semantically related to the global
knowledge represented in the retrieved community
summaries (i.e., broader, aggregated summaries),
as these individual entities may not be a part of the
retrieved communities for a query.
We highlight two critical challenges in exist-
ing RAG systems that integrate graph structures:
(1) distant structural relationship between se-
mantically similar entities and(2) knowledge
gap between local and global knowledge . We
illustrate them using a real example from a public
dataset, as shown in Figure 1.
Challenge (1) occurs because existing methods
over-rely on source documents, often resulting in
constructing a knowledge graph (KG) with many
entities that are not structurally proximate in the
KG even though they share semantically similar
attributes. For example, in Figure 1, although the
entities "BIG DATA" and "RECOMMENDATION
SYSTEM" share semantic relevance under the con-
cept of "DATA MINING", their distant structural
relationship in the KG reflects a corpus-driven dis-
connect. These inconsistencies between semantic
relevance and structural proximity are systemic in
KGs, undermining their utility in RAG systemsarXiv:2503.10150v1  [cs.CL]  13 Mar 2025

Figure 1: The challenges faced by existing RAG systems: (1) Distant structural relationship between semantically
similar entities. (2) Knowledge gap between local and global knowledge.
where contextual coherence is critical.
Challenge (2) occurs as existing methods (Guo
et al., 2024) (Edge et al., 2024) typically retrieve
context either from global or local perspectives but
fail to address the inherent disparity between these
knowledge layers. Consider the query "Please in-
troduce Amazon" in Figure 1, where global context
emphasizes Amazon’s involvement in technolog-
ical domains like big data and cloud computing,
but local context retrieves entities directly linked
to Amazon (e.g., subsidiaries, leadership). When
these two knowledge layers are fed into LLMs as
the contexts of a query without contextual align-
ment, LLMs may struggle to reconcile their distinct
scopes, leading to disjointed reasoning, incomplete
answers, or even contradictory outputs. For in-
stance, an LLM might conflate Amazon’s role as a
cloud provider (global) with its e-commerce oper-
ations (local), resulting in incoherent or factually
inconsistent responses as the red words shown in
the case. This underscores the need for new meth-
ods that bridge hierarchical knowledge layers to
ensure cohesive reasoning in RAG systems.
To address these challenges, we propose
Retrieval-Augmented Generation with Hierar-
chical Knowledge (HiRAG) , which integrates hier-
archical knowledge into the indexing and retrieval
processes. Hierarchical knowledge (Sarrafzadeh
and Lank, 2017) is a natural concept in both graph
structure and human cognition, yet it has been
overlooked in existing approaches. Specifically,
to address Challenge (1), we introduce Indexing
with Hierarchical Knowledge (HiIndex) . Rather
than simply constructing a flat KG, we index a
KG hierarchically layer by layer. Each entity (or
node) in a higher layer summarizes a cluster of
entities in the lower layer, which can enhancethe connectivity between semantically similar en-
tities. For example, in Figure 1, the inclusion of
the summary entity "DATA MINING" strengthens
the connection between "BIG DATA" and "REC-
OMMENDATION SYSTEM". To address Chal-
lenge (2), we propose Retrieval with Hierarchical
Knowledge (HiRetrieval) . HiRetrieval effectively
bridges local knowledge of entity descriptions to
global knowledge of communities, thus resolving
knowledge layer disparities. It provides a three-
level context comprising the global level, the bridge
level, and the local level knowledge to an LLM, en-
abling the LLM to generate more comprehensive
and precise responses.
In summary, we make the following main contri-
butions:
•We identify and address two critical chal-
lenges in graph-based RAG systems: distant
structural relationships between semantically
similar entities and the knowledge gap be-
tween local and global information.
•We propose HiRAG, which introduces unsu-
pervised hierarchical indexing and a novel
bridging mechanism for effective knowledge
integration, significantly advancing the state-
of-the-art in RAG systems.
•Extensive experiments demonstrate both the
effectiveness and efficiency of our approach,
with comprehensive ablation studies validat-
ing the contribution of each component.
2 Related Work
In this section, we discuss recent research con-
cerning graph-augmented LLMs, specifically RAG

methods with graph structures. GNN-RAG (Mavro-
matis and Karypis, 2024) employs GNN-based rea-
soning to retrieve query-related entities. Then they
find the shortest path between the retrieved entities
and candidate answer entities to construct reason-
ing paths. LightRAG (Guo et al., 2024) integrates
a dual-level retrieval method with graph-enhanced
text indexing. They also decrease the computa-
tional costs and speed up the adjustment process.
GRAG (Hu et al., 2024) leverages a soft pruning
approach to minimize the influence of irrelevant
entities in retrieved subgraphs. It also implements
prompt tuning to help LLMs comprehend textual
and topological information in subgraphs by in-
corporating graph soft prompts. StructRAG (Li
et al., 2024) identifies the most suitable structure
for each task, transforms the initial documents into
this organized structure, and subsequently gener-
ates responses according to the established struc-
ture. Microsoft GraphRAG (Edge et al., 2024)
first retrieves related communities and then let the
LLM generate the response with the retrieved com-
munities. They also answer a query with global
search and local search. KAG (Liang et al., 2024)
introduces a professional domain knowledge ser-
vice framework and employs knowledge alignment
using conceptual semantic reasoning to mitigate
the noise issue in OpenIE. KAG also constructs
domain expert knowledge using human-annotated
schemas.
3 Preliminary and Definitions
In this section, we give a general formulation of an
RAG system with graph structure referring to the
definitions in (Guo et al., 2024) and (Peng et al.,
2024b).
In an RAG framework Mas shown in Equa-
tion 1, LLM is the generation module, Rrepre-
sents the retrieval module, φdenotes the graph
indexer, and ψrefers to the graph retriever:
M= (LLM, R(φ, ψ)). (1)
When we answer a query, the answer we get from
an RAG system is represented by a∗, which can be
formulated as
a∗=argmax
a∈AM(a|q,G), (2)
G=φ(D) ={(h, r, t )|h, t∈ V, r∈ E},(3)
where M(a|q,G)is the target distribution with
a graph retriever ψ(G|q,G)and a generatorLLM (a|q, G), and Ais the set of possible re-
sponses. The graph database Gis constructed from
the original external database D. We utilize the
total probability formula to decompose M(a|q,G),
which can be expressed as
M(a|q,G) =X
G∈GLLM (a|q, G)·ψ(G|q,G).(4)
Most of the time, we only need to retrieve the
most relevant subgraph Gfrom the external graph
database G. Therefore, here we can approximate
M(a|q,G)as follows:
M(a|q,G)≈LLM (a|q, G∗)·ψ(G∗|q,G),(5)
where G∗denotes the optimal subgraph we retrieve
from the external graph database G. What we fi-
nally want is to get a better generated answer a∗.
4 The HiRAG Framework
HiRAG consists of the two modules, HiIndex and
HiRetrieval, as shown in Figure 2. In the HiIndex
module, we construct a hierarchical KG with differ-
ent knowledge granularity in different layers. The
summary entities in a higher layer represent more
coarse-grained, high-level knowledge but they can
enhance the connectivity between semantically sim-
ilar entities in a lower layer. In the HiRetrieval
module, we select the most relevant entities from
each retrieved community and find the shortest path
to connect them, which serve as the bridge-level
knowledge to connect the knowledge at both lo-
cal and global levels. Then an LLM will generate
responses with these three-level knowledge as the
context.
4.1 Indexing with Hierarchical Knowledge
In the HiIndex module, we index the input docu-
ments as a hierarchical KG. First, we employ the
entity-centric triple extraction to construct a basic
KGG0following (Carta et al., 2023). Specifically,
we split the input documents into text chunks with
some overlaps. These chunks will be fed into the
LLM with well-designed prompts to extract entities
V0first. Then the LLM will generate relations (or
edges) E0between pairs of the extracted entities
based on the information of the corresponding text
chunks. The basic KG can be represented as
G0={(h, r, t )|h, t∈ V0, r∈ E0}. (6)

Figure 2: The overall architecture of the HiRAG framework.
The basic KG is also the 0-th layer of our hierar-
chical KG. We denote the set of entities (nodes) in
thei-th layer as Liwhere L0=V0. To construct
thei-th layer of the hierarchical KG, for i≥1,
we first fetch the embeddings of the entities in the
(i−1)-th layer of the hierarchical KG, which is
denoted as
Zi−1={Embedding (v)|v∈ Li−1},(7)
where Embedding (v)is the embedding of an
entity v. Then we employ Gaussian Mixture Mod-
els (GMMs) to conduct semantical clustering on
Li−1based on Zi−1, following the method de-
scribed in RAPTOR (Sarthi et al., 2024). We obtain
a set of clusters as
Ci−1=GMM (Li−1,Zi−1) ={S1, . . . ,Sc},
(8)
where ∀x, y∈[1, c],Sx∩ S y=∅andS
1≤x≤cSx=Li−1. After clustering with GMMs,
the descriptions of the entities in each cluster in
Ci−1are fed into the LLM to generate a set of sum-
mary entities for the i-th layer. Thus, the set of sum-
mary entities in the i-th layer, i.e., Li, is the union
of the sets of summary entities generated from all
clusters in Ci−1. Then, we create the relations be-
tween entities in Li−1and entities in Li, denoted as
E{i−1,i}, by connecting the entities in each cluster
S ∈ C i−1to the corresponding summary entities in
Lithat are generated from the entities in S.
To generate summary entities in Li, we use a
set of meta summary entities Xto guide the LLMto generate the summary entities. Here, Xis a
small set of general concepts such as "organiza-
tion", "person", "location", "event", "technology",
etc., that are generated by LLM. For example, the
meta summary "technology" could guide the LLM
to generate summary entities such as "big data" and
"AI". Note that conceptually Xis added as the top
layer in Figure 2, but Xis actually not part of the
hierarchical KG.
After generating the summary entities and rela-
tions in the i-th layer, we update the KG as follows:
Ei=Ei−1∪ E{i−1,i}, (9)
Vi=Vi−1∪ Li, (10)
Gi={(h, r, t )|h, t∈ Vi, r∈ Ei}. (11)
We repeat the above process for each layer from
the 1st layer to the k-th layer. We will discuss
how to choose the parameter kin Section 5. Also
note that there is no relation between the summary
entities in each layer except the 0-th layer (i.e., the
basic KG).
We also employ the Leiden algorithm (Traag
et al., 2019) to compute a set of communities P
from the hierarchical KG. Each community may
contain entities from multiple layers and an entity
may appear in multiple communities. For each
community p∈ P, we generate an interpretable
semantic report using LLMs. Unlike existing meth-
ods such as GraphRAG (Edge et al., 2024) and

LightRAG (Guo et al., 2024), which identify com-
munities based solely on direct structural proxim-
ity in a basic KG, our hierarchical KG introduces
multi-resolution semantic aggregation. Higher-
layer entities in our KG act as semantic hubs that
abstract clusters of semantically related entities re-
gardless of their distance from each other in a lower
layer. For example, while a flat KG might sepa-
rate "cardiologist" and "neurologist" nodes due to
limited direct connections, their hierarchical ab-
straction as "medical specialists" in upper layers
enables joint community membership. The hierar-
chical structure thus provides dual connectivity en-
hancement: structural cohesion through localized
lower-layer connections and semantic bridging via
higher-layer abstractions. This dual mechanism
ensures our communities reflect both explicit re-
lational patterns and implicit conceptual relation-
ships, yielding more comprehensive knowledge
groupings than structure-only approaches.
4.2 Retrieval with Hierarchical Knowledge
We now discuss how we retrieve hierarchical
knowledge to address the knowledge gap issue.
Based on the hierarchical KG Gkconstructed in
Section 4.1, we retrieve three-level knowledge at
both local and global levels, as well as the bridging
knowledge that connects them.
To retrieve local-level knowledge, we extract the
top-nmost relevant entities ˆVas shown in Equa-
tion 12, where Sim(q, v)is a function that mea-
sures the semantic similarity between a user query
qand an entity vin the hierarchical KG Gk. We set
nto 20 as default.
ˆV=TopN ({v∈ Vk|Sim(q, v)}, n). (12)
To access global-level knowledge related to a query,
we find the communities ˆP ⊂ P that are con-
nected to the retrieved entities as described in Equa-
tion 13, where Pis computed during indexing in
Section 4.1. Then the community reports of these
communities are retrieved, which represent coarse-
grained knowledge relevant to the user’s query.
ˆP=[
p∈P{p|p∩ˆV ̸=∅}. (13)
To bridge the knowledge gap between the retrieved
local-level and global-level knowledge, we also
find a set of reasoning paths Rconnecting the re-
trieved communities. Specifically, from each com-
munity, we select the top- mquery-related key en-tities and collect them into ˆVˆP, as shown in Equa-
tion 14. The set of reasoning paths Ris defined as
the set of shortest paths between each pair of key
entities according to their order in ˆVˆP, as shown in
Equation 15. Based on R, we construct a subgraph
ˆRas described in Equation 16. Here, ˆRcollects a
set of triples from the KG that connect the knowl-
edge in the local entities and the knowledge in the
global communities.
ˆVˆP=[
p∈ˆPTopN ({v∈p|Sim(q, v)}, m),(14)
R=[
i∈[1,|ˆVˆP|−1]ShortestPath Gk(ˆVˆP[i],ˆVˆP[i+1]),
(15)
ˆR={(h, r, t )∈ Gk|h, t∈ R} . (16)
After retrieving the three-level hierarchical knowl-
edge, i.e., local-level descriptions of the individual
entities in ˆV, global-level community reports of the
communities in ˆP, and bridge-level descriptions of
the triples in ˆR, we feed them as the context to the
LLM to generate a comprehensive answer to the
query. We also provide the detailed procedures of
HiRAG with pseudocodes in Appendix C.
4.3 Why is HiRAG effective?
HiRAG’s efficacy stems from its hierarchical archi-
tecture, HiIndex (i.e., hierarchical KG) and HiRe-
trieval (i.e., three-level knowledge retrieval), which
directly mitigates the limitations outlined in Chal-
lenges (1) and (2) as described in Section 1.
Addressing Challenge (1): The hierarchical
knowledge graph Gkintroduces summary entities
in its higher layers, creating shortcuts between enti-
ties that are distantly located in lower layers. This
design bridges semantically related concepts effi-
ciently, bypassing the need for exhaustive traversal
of fine-grained relationships in the KG.
Resolving Challenge (2): HiRetrieval con-
structs reasoning paths by linking the top- nentities
most semantically relevant to a query with their
associated communities. These paths represent
the shortest connections between localized entity
descriptions and global community-level insights,
ensuring that both granular details and broader con-
textual knowledge inform the reasoning process.
Synthesis: By integrating (i) semantically sim-
ilar entities via hierarchical shortcuts, (ii) global
community contexts, and (iii) optimized pathways
connecting local and global knowledge, HiRAG

generates comprehensive, context-aware answers
to user queries.
5 Experimental Evaluation
We report the performance evaluation results of
HiRAG in this section.
Baseline Methods. We compared HiRAG with
state-of-the-art and popular baseline RAG meth-
ods. NaiveRAG (Gao et al., 2022) (Gao et al.,
2023) splits original documents into chunks and
retrieves relevant text chunks through vector search.
GraphRAG (Edge et al., 2024) utilizes commu-
nities and we use the local search mode in our
experiments as it retrieves community reports as
global knowledge, while their global search mode
is known to be too costly and does not use local
entity descriptions. LightRAG (Guo et al., 2024)
uses both global and local knowledge to answer a
query. FastGraphRAG (Circlemind, 2024) inte-
grates KG and personalized PageRank as proposed
in HippoRAG (Gutiérrez et al., 2025). KAG (Liang
et al., 2024) integrates structured reasoning of
KG with LLMs and employs mutual indexing and
logical-form-guided reasoning to enhance profes-
sional domain knowledge services.
Datasets and Queries. We used four datasets
from the UltraDomain benchmark (Qian et al.,
2024), which is designed to evaluate RAG sys-
tems across diverse applications, focusing on long-
context tasks and high-level queries in specialized
domains. We used Mix, CS, Legal, and Agriculture
datasets like in LightRAG (Guo et al., 2024). We
also used the benchmark queries provided in Ultra-
Domain for each of the four datasets. The statistics
of these datasets are given in Appendix A.
LLM. We employed DeepSeek-V3 (DeepSeek-
AI et al., 2024) as the LLM for information extrac-
tion, entity summarization, and answer generation
in HiRAG and other baseline methods. We utilized
GLM-4-Plus (GLM et al., 2024) as the embedding
model for vector search and semantic clustering be-
cause DeepSeek-V3 does not provide an accessible
embedding model.
5.1 Overall Performance Comparison
Evaluation Details. Our experiments followed the
evaluation methods of recent work (Edge et al.,
2024)(Guo et al., 2024) by employing a power-
ful LLM to conduct multi-dimensional comparison.
We used the win rate to compare different methods,
which indicates the percentage of instances thata method generates higher-quality answers com-
pared to another method as judged by the LLM.
We utilized GPT-4o (Achiam et al., 2023) as the
evaluation model to judge which method generates
a superior answer for each query for the following
four dimensions: (1) Comprehensiveness : how
thoroughly does the answer address the question,
covering all relevant aspects and details? (2) Em-
powerment : how effectively does the answer pro-
vide actionable insights or solutions that empower
the user to take meaningful steps? (3) Diversity :
how well does the answer incorporate a variety of
perspectives, approaches, or solutions to the prob-
lem? (4) Overall : how does the answer perform
overall, considering comprehensiveness, empower-
ment, diversity, and any other relevant factors? For
a fair comparison, we also alternated the order of
the answers generated by each pair of methods in
the prompts and calculated the overall win rates of
each method.
Evaluation Results. We report the win rates of
HiRAG and the five baseline methods in Table 1.
HiRAG outperforms the baselines accross the four
datasets and the four dimensions in most of the
cases. Here are the conclusions we can draw from
the results:
Evaluation Results. We present the win rates of
HiRAG and five baseline methods in Table 1. Hi-
RAG consistently outperforms existing approaches
across all four datasets and four evaluation dimen-
sions in the majority of cases. Key insights derived
from the results are summarized below.
Graph structure enhances RAG systems:
NaiveRAG exhibits inferior performance com-
pared to methods integrating graph structures,
primarily due to its inability to model relationships
between entities in retrieved components. Fur-
thermore, its context processing is constrained by
the token limitations of LLMs, highlighting the
importance of structured knowledge representation
for robust retrieval and reasoning.
Global knowledge improves answer qual-
ity:Approaches incorporating global knowledge
(GraphRAG, LightRAG, KAG, HiRAG) signifi-
cantly surpass FastGraphRAG, which relies on lo-
cal knowledge via personalized PageRank. An-
swers generated without global context lack depth
and diversity, underscoring the necessity of holis-
tic knowledge integration for comprehensive re-
sponses.
HiRAG’s superior performance: Among graph-
enhanced RAG systems, HiRAG achieves the high-

Table 1: Win rates (%) of HiRAG, its two variants (for ablation study), and baseline methods.
Mix CS Legal Agriculture
NaiveRAG HiRAG NaiveRAG HiRAG NaiveRAG HiRAG NaiveRAG HiRAG
Comprehensiveness 16.6% 83.4% 30.0% 70.0% 32.5% 67.5% 34.0% 66.0%
Empowerment 11.6% 88.4% 29.0% 71.0% 25.0% 75.0% 31.0% 69.0%
Diversity 12.7% 87.3% 14.5% 85.5% 22.0% 78.0% 21.0% 79.0%
Overall 12.4% 87.6% 26.5% 73.5% 25.5% 74.5% 28.5% 71.5%
GraphRAG HiRAG GraphRAG HiRAG GraphRAG HiRAG GraphRAG HiRAG
Comprehensiveness 42.1% 57.9% 40.5% 59.5% 48.5% 51.5% 49.0% 51.0%
Empowerment 35.1% 64.9% 38.5% 61.5% 43.5% 56.5% 48.5% 51.5%
Diversity 40.5% 59.5% 30.5% 69.5% 47.0% 53.0% 45.5% 54.5%
Overall 35.9% 64.1% 36.0% 64.0% 45.5% 54.5% 46.0% 54.0%
LightRAG HiRAG LightRAG HiRAG LightRAG HiRAG LightRAG HiRAG
Comprehensiveness 36.8% 63.2% 44.5% 55.5% 49.0% 51.0% 38.5% 61.5%
Empowerment 34.9% 65.1% 41.5% 58.5% 43.5% 56.5% 36.5% 63.5%
Diversity 34.1% 65.9% 33.0% 67.0% 63.0% 37.0% 37.5% 62.5%
Overall 34.1% 65.9% 41.0% 59.0% 48.0% 52.0% 38.5% 61.5%
FastGraphRAG HiRAG FastGraphRAG HiRAG FastGraphRAG HiRAG FastGraphRAG HiRAG
Comprehensiveness 0.8% 99.2% 0.0% 100.0% 1.0% 99.0% 0.0% 100.0%
Empowerment 0.8% 99.2% 0.0% 100.0% 0.0% 100.0% 0.0% 100.0%
Diversity 0.8% 99.2% 0.5% 99.5% 1.5% 98.5% 0.0% 100.0%
Overall 0.8% 99.2% 0.0% 100.0% 0.0% 100.0% 0.0% 100.0%
KAG HiRAG KAG HiRAG KAG HiRAG KAG HiRAG
Comprehensiveness 2.3% 97.7% 1.0% 99.0% 16.5% 83.5% 5.0% 99.5%
Empowerment 3.5% 96.5% 4.5% 95.5% 9.0% 91.0% 5.0% 99.5%
Diversity 3.8% 96.2% 5.0% 95.0% 11.0% 89.0% 3.5% 96.5%
Overall 2.3% 97.7% 1.5% 98.5% 8.5% 91.5% 0.0% 100.0%
w/o HiIndex HiRAG w/o HiIndex HiRAG w/o HiIndex HiRAG w/o HiIndex HiRAG
Comprehensiveness 46.7% 53.3% 44.2% 55.8% 49.0% 51.0% 50.5% 49.5%
Empowerment 43.2% 56.8% 38.8% 61.2% 47.5% 52.5% 50.5% 49.5%
Diversity 40.5% 59.5% 40.0% 60.0% 48.0% 52.0% 48.5% 51.5%
Overall 42.4% 57.6% 40.0% 60.0% 46.5% 53.5% 48.0% 52.0%
w/o Bridge HiRAG w/o Bridge HiRAG w/o Bridge HiRAG w/o Bridge HiRAG
Comprehensiveness 49.2% 50.8% 46.5% 53.5% 49.5% 50.5% 47.0% 53.0%
Empowerment 44.2% 55.8% 43.0% 57.0% 38.5% 61.5% 41.0% 59.0%
Diversity 44.6% 55.4% 44.0% 56.0% 43.5% 56.5% 46.0% 54.0%
Overall 47.3% 52.7% 42.5% 57.5% 44.0% 56.0% 42.0% 58.0%
est performance across all datasets (spanning di-
verse domains) and evaluation dimensions. This
superiority stems primarily from two innovations:
(1) HiIndex which enhances connections between
remote but semantically similar entities in the hier-
archical KG, and (2) HiRetrieval which effectively
bridges global knowledge with localized context to
optimize relevance and coherence.
5.2 Hierarchical KG vs. Flat KG
To evaluate the significance of the hierarchical KG,
we replace the hierarchical KG with a flat KG (or
a basic KG), denoted by w/o HiIndex as reported
in Table 1. Compared with HiRAG, the win rates
ofw/o HiIndex drop in almost all cases and quite
significantly in at least half of the cases. This abla-
tion study thus shows that the hierarchical indexingplays an important role in the quality of answer gen-
eration, since the connectivity among semantically
similar entities is enhanced with the hierarchical
KG, with which related entities can be grouped
together both from structural and semantical per-
spectives.
From Table 1, we also observe that the win rates
ofw/o HiIndex are better or comparable to those
of GraphRAG and LightRAG when compared with
HiRAG. This suggests that our three-level knowl-
edge retrieval method, i.e., HiRetrieval, is effective
even applied on a flat KG, because GraphRAG and
LightRAG also index on a flat KG but they only
use the local entity descriptions and global commu-
nity reports, while w/o HiIndex uses an additional
bridge-level knowledge.

Table 2: Comparisons in terms of tokens, API calls and time cost across four datasets.
Token Cost API Calls Time Cost (s)
Dataset Method Indexing Retrieval Indexing Retrieval Indexing Retrieval
MixGraphRAG 8,507,697 0.00 2,666 1.00 6,696 0.70
LightRAG 3,849,030 357.76 1,160 2.00 3,342 3.06
KAG 6,440,668 110,532.00 831 9.17 8,530 58.47
HiRAG 21,898,765 0.00 6,790 1.00 17,208 0.85
CSGraphRAG 27,506,689 0.00 8,649 1.00 19,255 0.98
LightRAG 12,638,997 353.37 3,799 2.00 14,307 4.97
KAG 7,358,717 89,746.00 2,190 6.29 14,837 46.37
HiRAG 56,042,906 0.00 16,535 1.00 44,994 1.17
LegalGraphRAG 51,168,359 0.00 13,560 1.00 30,065 1.12
LightRAG 30,299,958 353.77 9,442 2.00 21,505 5.44
KAG 18,431,706 97,683.00 4,980 7.82 29,191 51.26
HiRAG 106,427,778 0.00 27,224 1.00 115,232 2.04
AgricultureGraphRAG 27,974,472 0.00 8,669 1.00 20,362 1.17
LightRAG 12,031,096 354.62 3,694 2.00 13,550 5.64
KAG 7,513,424 93,217.00 2,358 6.83 22,557 49.57
HiRAG 96,080,883 0.00 22,736 1.00 50,920 1.76
Figure 3: Answer to the query in Figure 1 with addi-
tional bridge-level knowledge.
5.3 HiRetrieval vs. Gapped Knowledge
To show the effectiveness of HiRetrieval, we also
created another variant of HiRAG without using the
bridge-level knowledge, denoted by w/o Bridge in
Table 1. The result shows that without the bridge-
layer knowledge, the win rates drop significantly
across all datasets and evaluation dimensions, be-
cause there is knowledge gap between the local-
level and global-level knowledge as discussed in
Section 1.
Case Study. Figure 3 shows the three-level
knowledge used as the context to an LLM to answer
the query in Figure 1. The bridge-level knowledge
contains entity descriptions from different commu-
nities, as shown by the different colors in Figure 3,
which helps the LLM correctly answer the question
about Amazon’s role as an e-commerce and cloud
provider.
Figure 4: Cluster sparsity CS iand change rate from
CS itoCS i+1, where the shadow areas represent the
value ranges of the four datasets and the blue/pink lines
are the respective average values.
5.4 Determining the Number of Layers
One important thing in HiIndex is to determine the
number of layers, k, for the hierarchical KG, which
should be determined dynamically according to the
quality of clusters in each layer. We stop build-
ing another layer when the majority of the clusters
consist of only a small number of entities, mean-
ing that the entities can no longer be effectively
grouped together. To measure that, we introduce
the notion of cluster sparsity CSi, as inspired by
graph sparsity, to measure the quality of clusters in
thei-th layer as described in Equation 17.
CSi= 1−P
S∈C i|S|(|S| − 1)
|Li|(|Li| −1). (17)
The more the clusters in Cihave a small number
of entities, the larger is CSi, where the worst case
is when each cluster contains only one entity (i.e.,
CSi= 1). Figure 4 shows that as we have more
layers, the cluster sparsity increases and then sta-

bilizes. We also plot the change rate from CSi
toCSi+1, which shows that there is little or no
more change after constructing a certain number
of layers. We set a threshold ϵ= 5% and stop
constructing another layer when the change rate of
cluster sparsity is lower than ϵbecause the cluster
quality has little or no improvement after that.
5.5 Efficiency and Costs Analysis
To evaluate the efficiency and costs of HiRAG,
we also report the token costs, the number of API
calls, and the time costs of indexing and retrieval
of HiRAG and the baselines in Table 2. For index-
ing, we record the total costs of the entire process.
Although HiRAG needs more time and resources
to conduct indexing for better performance, we
remark that indexing is offline and the total cost
is only about 7.55 USD for the Mix dataset us-
ing DeepSeek-V3. In terms of retrieval, we calcu-
late the average costs per query. Unlike KAG and
LightRAG, HiRAG does not cost any tokens for
retrieval. Therefore, HiRAG is more efficient for
online retrieval.
6 Conclusions
We presented a new approach to enhance RAG sys-
tems by effectively utilizing graph structures with
hierarchical knowledge. By developing (1) HiIn-
dex which enhances structural and semantic con-
nectivity across hierarchical layers, and (2) HiRe-
trieval which effectively bridges global conceptual
abstractions with localized entity descriptions, Hi-
RAG achieves superior performance than existing
methods.
7 Limitations
HiRAG has the following limitations. Firstly, con-
structing a high-quality hierarchical KG may incur
substantial token consumption and time overhead,
as LLMs need to perform entity summarization in
each layer. However, the monetary cost of using
LLMs may not be the major concern as the cost
is decreasing rapidly recently, and therefore we
may consider parallelizing the indexing process to
reduce the indexing time. Secondly, the retrieval
module requires more sophisticated query-aware
ranking mechanisms. Currently, our HiRetrieval
module relies solely on LLM-generated weights
for relation ranking, which may affect query rele-
vance. We will research for more effective ranking
mechanisms to further improve the retrieval quality.References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, et al. 2023. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774 .
Salvatore Carta, Alessandro Giuliani, Leonardo Piano,
Alessandro Sebastian Podda, Livio Pompianu, and
Sandro Gabriele Tiddia. 2023. Iterative zero-shot llm
prompting for knowledge graph construction. arXiv
preprint arXiv:2307.01128 .
Circlemind. 2024. fast-graphrag. https://github.
com/circlemind-ai/fast-graphrag .
DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingx-
uan Wang, Bochao Wu, Chengda Lu, Chenggang
Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
Damai Dai, Daya Guo, Dejian Yang, Deli Chen,
Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai,
Fuli Luo, Guangbo Hao, Guanting Chen, Guowei
Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng
Wang, Haowei Zhang, Honghui Ding, Huajian Xin,
Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang,
Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang,
Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie
Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu,
Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean
Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao,
Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang,
Mingchuan Zhang, Minghua Zhang, Minghui Tang,
Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang,
Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu
Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge,
Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin
Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao
Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu,
Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu
Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou,
Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun,
W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An,
Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu,
Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang,
Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen,
Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen,
Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin
Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu,
Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang,
Xinyuan Li, Xuecheng Su, Xuheng Lin, Y . K. Li,
Y . Q. Wang, Y . X. Wei, Y . X. Zhu, Yang Zhang, Yan-
hong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao
Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu,
Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong,
Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yix-
uan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo,
Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue
Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan
Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxi-
ang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z.
Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu,
Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan
Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhi-
gang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu,

Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu,
Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi
Gao, and Zizheng Pan. 2024. Deepseek-v3 technical
report. Preprint , arXiv:2412.19437.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130 .
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. In Pro-
ceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining , pages 6491–
6501.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2022. Precise zero-shot dense retrieval without rele-
vance labels. arXiv preprint arXiv:2212.10496 .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2023. Retrieval-augmented gen-
eration for large language models: A survey. arXiv
preprint arXiv:2312.10997 .
Team GLM, :, Aohan Zeng, Bin Xu, Bowen Wang,
Chenhui Zhang, Da Yin, Dan Zhang, Diego Ro-
jas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao
Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale
Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Jingyu Sun,
Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong,
Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai
Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin
Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao,
Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin
Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan
Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu,
Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong,
Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du,
Zhenyu Hou, and Zihan Wang. 2024. Chatglm: A
family of large language models from glm-130b to
glm-4 all tools. Preprint , arXiv:2406.12793.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and
Chao Huang. 2024. Lightrag: Simple and fast
retrieval-augmented generation. arXiv preprint
arXiv:2410.05779 .
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. Hipporag: Neu-
robiologically inspired long-term memory for large
language models. In The Thirty-eighth Annual Con-
ference on Neural Information Processing Systems .
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2025. Hipporag: Neu-
robiologically inspired long-term memory for large
language models. Preprint , arXiv:2405.14831.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. arXiv preprint arXiv:2011.01060 .Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan,
Chen Ling, and Liang Zhao. 2024. Grag: Graph
retrieval-augmented generation. arXiv preprint
arXiv:2405.16506 .
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu
Lin, Yaojie Lu, Qiaoyu Tang, Fei Huang, Xian-
pei Han, Le Sun, and Yongbin Li. 2024. Struc-
trag: Boosting knowledge intensive reasoning of llms
via inference-time hybrid information structurization.
arXiv preprint arXiv:2410.08815 .
Lei Liang, Mengshu Sun, Zhengke Gui, Zhongshu
Zhu, Zhouyu Jiang, Ling Zhong, Yuan Qu, Pei-
long Zhao, Zhongpu Bo, Jin Yang, Huaidong Xiong,
Lin Yuan, Jun Xu, Zaoyang Wang, Zhiqiang Zhang,
Wen Zhang, Huajun Chen, Wenguang Chen, and
Jun Zhou. 2024. Kag: Boosting llms in profes-
sional domains via knowledge augmented generation.
Preprint , arXiv:2409.13731.
Costas Mavromatis and George Karypis. 2024. Gnn-
rag: Graph neural retrieval for large language model
reasoning. arXiv preprint arXiv:2405.20139 .
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024a. Graph retrieval-augmented generation:
A survey. Preprint , arXiv:2408.08921.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024b. Graph retrieval-augmented generation:
A survey. arXiv preprint arXiv:2408.08921 .
Hongjin Qian, Peitian Zhang, Zheng Liu, Kelong Mao,
and Zhicheng Dou. 2024. Memorag: Moving to-
wards next-gen rag via memory-inspired knowledge
discovery. arXiv preprint arXiv:2409.05591 .
Bahareh Sarrafzadeh and Edward Lank. 2017. Improv-
ing exploratory search experience through hierarchi-
cal knowledge graphs. In Proceedings of the 40th
international ACM SIGIR conference on research
and development in information retrieval , pages 145–
154.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Man-
ning. 2024. Raptor: Recursive abstractive pro-
cessing for tree-organized retrieval. arXiv preprint
arXiv:2401.18059 .
Yixuan Tang and Yi Yang. 2024. Multihop-rag: Bench-
marking retrieval-augmented generation for multi-
hop queries. Preprint , arXiv:2401.15391.

Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck.
2019. From louvain to leiden: guaranteeing well-
connected communities. Scientific reports , 9(1):1–
12.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. arXiv preprint arXiv:1809.09600 .
Qinggang Zhang, Shengyuan Chen, Yuanchen Bei,
Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan
Dong, Hao Chen, Yi Chang, and Xiao Huang. 2025.
A survey of graph retrieval-augmented generation
for customized large language models. Preprint ,
arXiv:2501.13958.
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu,
Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang,
Yulong Chen, et al. 2023. Siren’s song in the ai ocean:
a survey on hallucination in large language models.
arXiv preprint arXiv:2309.01219 .
Appendix
In this section, we delve into the construction of the
hierarchical KG with the HiIndex module, accom-
panied by illustrative pseudo-codes. We present
statistics and a simple case study to demonstrate
the enhanced connectivity among entities in the hi-
erarchical KG. Additionally, we give well designed
prompt templates used in HiRAG.
A Experimental Datasets
Table 3: Statistics of datasets.
Dataset Mix CS Legal Agriculture
# of Documents 61 10 94 12
# of Tokens 625948 2210894 5279400 2028496
Table 3 presents the statistical characteristics
of the experimental datasets, where all documents
were consistently tokenized using Byte Pair Encod-
ing (BPE) tokenizer "cl100k_base".
B Evaluations with Objective Metrics
To objectively evaluate the QA performance of
HiRAG and the baseline methods, we leverage
two established metrics: exact match ( EM) and
F1scores, applied to the generated answers. We
perform systematic evaluations using GPT-4o-mini
on two multi-hop QA datasets: HotpotQA (Yang
et al., 2018) and 2WikiMultiHopQA (Ho et al.,
2020). For a consistent comparison with previous
work, we follow the settings of HippoRAG (Gutiér-
rez et al., 2024), obtaining 1,000 queries from eachTable 4: QA performances of HiRAG and other baseline
methods with EM and F1 scores.
2WikiMultiHopQA HotpotQA
Method EM (%) F1 (%) EM (%) F1 (%)
NaiveRAG 15.60 25.64 21.60 40.19
GraphRAG 22.50 27.49 31.70 42.74
LightRAG 16.50 40.95 25.00 43.20
FastGraphRAG 20.80 44.81 35.00 49.56
HiRAG 46.20 60.06 37.00 52.29
validation set. We did not present the results of
KAG because, despite our efforts to implement
it, we were unable to make it fully work on this
benchmark.
Compared with the metric of win rates, the per-
formances with EM and F1 scores can indicate
HiRAG’s ability to achieve correctness. Given that
the RAG system has access to richer contexts, it
tends to produce more comprehensive responses.
Nevertheless, while comprehensiveness, empower-
ment, and diversity are important qualities for the
generated answers, correctness is equally essential.
As illustrated in Table 4, HiRAG is also capable of
generating more accurate answers compared to the
baseline methods.
C Implementation Details of HiRAG
We give a more detailed and formulated expres-
sion of hierarchical indexing (HiIndex) and hier-
archical retrieval (HiRetrieval). As described in
Algorithm 1, the hierarchical knowledge graph is
constructed iteratively. The number of clustered
layers depends on the rate of change in the cluster
sparsity at each layer. As shown in Algorithm 2,
we retrieve knowledge of three layers (local layer,
global layer, and bridge layer) as contexts for LLM
to generate more comprehensive and accurate an-
swers.
D The Clustering Coefficients of HiIndex
We calculate and compare the clustering coeffi-
cients of GraphRAG, LightRAG and HiRAG in Fig-
ure 5. HiRAG shows a higher clustering coefficient
than other baseline methods, which means that
more entities in the hierarchical KG constructed by
the HiIndex module tend to cluster together. And
this is also the reason why the HiIndex module can
improve the performance of RAG systems.

Algorithm 1: HiIndex
Input: Basic knowledge graph G0extracted by
the LLM; Predefined threshold ϵ;
Output: Hierarchical knowledge graph Gk;
1:L0← V 0;
2:Z0← {Embedding (v)|v∈ L0};
3:i←1;
4:while True do
5: /*Perform semantical clustering*/
6:Ci−1←GMM (Gi−1,Zi−1);
7: /*Calculate cluster sparsity*/
8:CSi←1−P
S∈Ci−1|S|(|S|−1)
|Li−1|(|Li−1|−1);
9: ifchange rate of CSi≤ϵthen
10: i←i−1;
11: break;
12: end if
13: /*Generate summary entities and
relations*/
14:Li← {} ;
15:E{i−1,i}← {} ;
16: forSxinCi−1do
17: L,E ←LLM (Sx,X);
18: Li← L i∪ L;
19: E{i−1,i}← E{i−1,i}∪ E;
20: end for
21:Zi={Embedding (v)|v∈ Li};
22: /*Update KG*/
23:Ei← E i−1∪ E{i−1,i};
24:Vi← V i−1∪ Li;
25:Gi← {(h, r, t )|h, t∈ Vi, r∈ Ei}
26: i←i+ 1;
27:end while
28:k←i;
29:Gk← {(h, r, t )|h, t∈ Vk, r∈ Ek};
E A Simple Case of Hierarchical KG
As shown in Figure 6, we fix the issues mentioned
in Section 1 with a hierarchical KG. This case
demonstrates that the GMMs clustered semanti-
cally similar entities "BIG DATA" and "RECOM-
MENDATION SYSTEM" together. The LLM sum-
marizes "DISTRIBUTED COMPUTING" as their
shared summary entities in the next layer. As a con-
sequence, the connections between these related
entities can be enhanced from a semantic perspec-
tive.Algorithm 2: HiRetrieval
Input: The hierarchical knowledge graph Gk; The
detected community set PinGk; The number
of retrieved entities n; The number of selected
key entities min each retrieved community;
Output: The generated answer a;
1:/*The local-layer knowledge context*/
2:ˆV ← TopN ({v∈ Vk|Sim(v, q)}, n);
3:/*The global-layer knowledge context*/
4:ˆP ←S
p∈P{p|p∩ˆV ̸=ϕ};
5:ˆR ← {} ;
6:ˆVˆP← {} ;
7:/*Select key entities*/
8:forpinˆPdo
9: ˆVˆP←ˆVˆP∪TopN ({v∈
p|Sim(v, q)}, m);
10:end for
11:/*Find the reasoning path*/
12:foriin[1,|ˆVˆP| −1]do
13:R ←
R ∪ShortestPath Gk(ˆVˆP[i],ˆVˆP[i+ 1]) ;
14:end for
15:/*The bridge-layer knowledge context*/
16:ˆR ← { (h, r, t )∈ Gk|h, t∈ R} ;
17:/*Generate the answer*/
18:a←LLM (q,ˆV,ˆR,ˆP);
F Prompt Templates used in HiRAG
F.1 Prompt Templates for Entity Extraction
As shown in Figure 7, we used that prompt template
to extract entities from text chunks. We also give
three examples to guide the LLM to extract entities
with higher accuracy.
F.2 Prompt Templates for Relation Extraction
As shown in Figure 8, we extract relations from the
entities extracted earlier and the corresponding text
Figure 5: Comparisons between the clustering coeffi-
cients of GraphRAG, LightRAG and HiRAG across four
datasets.

Figure 6: The shortest path with hierarchical KG be-
tween the entities in the case mentioned in the introduc-
tion.
chunks. Then we can get the triples in the basic
knowledge graph, which is also the 0-th layer of
the hierarchical knowledge graph.
F.3 Prompt Templates for Entity
Summarization
As shown in Figure 9, we generate summary en-
tities in each layer of the hierarchical knowledgegraph. We will not only let the LLM generate the
summary entities from the previous layer, but also
let it generate the relations between the entities of
these two layers. These relations will clarify the
reasons for summarizing these entities.
F.4 Prompt Templates for RAG Evaluation
In terms of the prompt templates we use to conduct
evaluations, we utilize the same prompt design as
that in LightRAG. The prompt will let the LLM
generate both evaluation results and the reasons in
JSON format to ensure clarity and accuracy.

Figure 7: The prompt template designed to extract entities from text chunks.

Figure 8: The prompt template designed to extract relations from entities and text chunks.

Figure 9: The prompt template designed to generate summary entities and the corresponding relations.