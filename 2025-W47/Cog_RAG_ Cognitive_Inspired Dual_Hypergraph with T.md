# Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation

**Authors**: Hao Hu, Yifan Feng, Ruoxue Li, Rundong Xue, Xingliang Hou, Zhiqiang Tian, Yue Gao, Shaoyi Du

**Published**: 2025-11-17 10:10:33

**PDF URL**: [https://arxiv.org/pdf/2511.13201v1](https://arxiv.org/pdf/2511.13201v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances the response quality and domain-specific performance of large language models (LLMs) by incorporating external knowledge to combat hallucinations. In recent research, graph structures have been integrated into RAG to enhance the capture of semantic relations between entities. However, it primarily focuses on low-order pairwise entity relations, limiting the high-order associations among multiple entities. Hypergraph-enhanced approaches address this limitation by modeling multi-entity interactions via hyperedges, but they are typically constrained to inter-chunk entity-level representations, overlooking the global thematic organization and alignment across chunks. Drawing inspiration from the top-down cognitive process of human reasoning, we propose a theme-aligned dual-hypergraph RAG framework (Cog-RAG) that uses a theme hypergraph to capture inter-chunk thematic structure and an entity hypergraph to model high-order semantic relations. Furthermore, we design a cognitive-inspired two-stage retrieval strategy that first activates query-relevant thematic content from the theme hypergraph, and then guides fine-grained recall and diffusion in the entity hypergraph, achieving semantic alignment and consistent generation from global themes to local details. Our extensive experiments demonstrate that Cog-RAG significantly outperforms existing state-of-the-art baseline approaches.

## Full Text


<!-- PDF content starts -->

Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment
Retrieval-Augmented Generation
Hao Hu1, Yifan Feng2, Ruoxue Li3, Rundong Xue1, Xingliang Hou4,
Zhiqiang Tian4, Yue Gao2, Shaoyi Du1*,
1State Key Laboratory of Human-Machine Hybrid Augmented Intelligence, National Engineering Research Center for Visual
Information and Applications, and Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University
2BNRist, THUIBCS, BLBCI, School of Software, Tsinghua University
3School of Artificial Intelligence, Xidian University
4School of Software Engineering, Xi’an Jiaotong University
huhao@stu.xjtu.edu.cn, dushaoyi@xjtu.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) enhances the re-
sponse quality and domain-specific performance of large lan-
guage models (LLMs) by incorporating external knowledge
to combat hallucinations. In recent research, graph struc-
tures have been integrated into RAG to enhance the capture
of semantic relations between entities. However, it primarily
focuses on low-order pairwise entity relations, limiting the
high-order associations among multiple entities. Hypergraph-
enhanced approaches address this limitation by modeling
multi-entity interactions via hyperedges, but they are typi-
cally constrained to inter-chunk entity-level representations,
overlooking the global thematic organization and alignment
across chunks. Drawing inspiration from the top-down cogni-
tive process of human reasoning, we propose a theme-aligned
dual-hypergraph RAG framework (Cog-RAG) that uses a
theme hypergraph to capture inter-chunk thematic structure
and an entity hypergraph to model high-order semantic rela-
tions. Furthermore, we design a cognitive-inspired two-stage
retrieval strategy that first activates query-relevant thematic
content from the theme hypergraph, and then guides fine-
grained recall and diffusion in the entity hypergraph, achiev-
ing semantic alignment and consistent generation from global
themes to local details. Our extensive experiments demon-
strate that Cog-RAG significantly outperforms existing state-
of-the-art baseline approaches.
Introduction
Retrieval-Augmented Generation (RAG) has recently
gained increasing attention for enhancing the performance
of large language models (LLMs) on knowledge-intensive
tasks (Lewis et al. 2020; Gao et al. 2023; Li et al. 2024).
It combats LLMs’ hallucination by incorporating external
knowledge, thereby enhancing response quality and reliabil-
ity (Ayala and Bechard 2024; Xia et al. 2025). Moreover, it
enables integration with private or domain-specific knowl-
edge bases, thereby increasing the model’s adaptability to
vertical domains. With these advantages, RAG has emerged
as a fundamental component in question answering, doc-
*Corresponding Authors.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
Figure 1: Knowledge modeling of graph, hypergraph, and
our theme-enhanced RAG.
ument understanding, and intelligent assistants (Fan et al.
2024; Dong et al. 2025).
Despite the notable potential of RAG in enhancing LLMs’
response quality, current methods mostly rely on a flattened
chunk-based retrieval that matches queries to document
chunks via vector similarity (Asai et al. 2023; Yang et al.
2024). However, this fails to capture inter-chunk depen-
dencies and semantic hierarchies, resulting in fragmented
and weakly connected retrieval content, which weakens the
model’s structured understanding of the entire knowledge.
To address this, recent studies have attempted to introduce
graph structures into RAG framework, aiming to construct
corpus-wide knowledge graphs that capture the structural se-
mantic relations between entities (Peng et al. 2024; Zhang
et al. 2025; Wang et al. 2025). For instance, GraphRAG
(Edge et al. 2024) and LightRAG (Guo et al. 2024) utilize
graph structures to strengthen entity-level indexing and re-
trieval, explicitly capturing semantic relations to improve in-
formation organization. Hyper-RAG (Feng et al. 2025a) uses
hypergraphs to model complex relations between multiple
entities. Nevertheless, these approaches primarily concen-
trate on entity-level structural modeling and lack a unified
organization of knowledge themes and semantic-driven rea-
soning, making it difficult to support hierarchical integrationarXiv:2511.13201v1  [cs.IR]  17 Nov 2025

of information from macro comprehension to micro details.
It is worth noting that humans tend to follow a top-down
information processing path when handling complex tasks
(Cheng et al. 2025; Guti ´errez et al. 2024). They begin by
identifying the core themes of the problem and constructing
a global semantic scaffold. Based on this, they recall and in-
tegrate relevant details to form a coherent and structured re-
sponse. This “theme-driven, detail-recall” cognitive pattern
reflects the inherent hierarchical organization and semantic
coherence in human information processing.
Inspired by this cognitive insight, we propose a dual-
hypergraph with theme alignment RAG framework (Cog-
RAG). Figure 1 shows its difference with other methods
in knowledge modeling. Our method leverages a dual-
hypergraph structure to model the global theme structure
and fine-grained high-order semantic relations. In addition,
it introduces a cognitive-inspired two-stage retrieval strat-
egy that simulates the human top-down information com-
prehension process, thereby enhancing the semantic consis-
tency and structural expressiveness of generated responses.
The main contributions are summarized as follows:
• We propose Cog-RAG that simulates the human top-
down information processing path, enabling hierarchical
generation modeling from macro-level semantic compre-
hension to micro-level information integration.
• We design a dual-hypergraph semantic indexing scheme
to separately model global inter-chunk theme struc-
ture and intra-chunk fine-grained high-order semantic
relations, overcoming the limitations of prior graph-
enhanced RAG models that focus only on pairwise re-
lations and lack unified thematic organization.
• We develop a cognitive-inspired two-stage retrieval strat-
egy that first activates relevant context in the theme hy-
pergraph and then triggers detail recall and diffusion in
the entity hypergraph. This “theme-driven, detail-recall”
process enables semantic alignment across granularity
and significantly improves the coherence and quality of
the response.
Related Work
RAG with Knowledge Graph
Most text-based RAG methods (Asai et al. 2023; Zhang et al.
2024; Xia et al. 2025; Yang et al. 2024) rely on a flattened
paragraph structure, which makes it difficult to model se-
mantic associations and contextual dependencies across text
chunks, thereby limiting the accuracy and completeness of
generated responses. To address this issue, recent studies
(Sarmah et al. 2024; Peng et al. 2024) have explored knowl-
edge graphs within the RAG framework to structurally rep-
resent entities and relations, aiming to enhance the organi-
zation and semantic expressiveness of retrieved content.
Some recent studies (Guti ´errez et al. 2024; Li and Du
2023; Cheng et al. 2025) attempt to automatically ex-
tract knowledge graph triples from the corpus and retrieve
relevant subgraphs to improve content relevance and in-
terpretability. However, these methods typically construct
sparse graph structures, making it difficult to capture the fullsemantic space and contextual dependencies. To address the
semantic sparsity issue and better model the semantic struc-
ture of documents, graph-enhanced RAG approaches (Edge
et al. 2024; Guo et al. 2024; Chen et al. 2025) extract entities
and their relations, and directly build document-level graph
databases enriched with contextual information, thereby re-
ducing information loss during the text-to-graph conversion
process. GraphRAG and LightRAG employ LLMs to ex-
tract entities and relations from texts as vertices and edges
of the graph. Nevertheless, existing methods primarily focus
on low-order pairwise relations between entities, neglecting
high-order group associations and global topic modeling,
which limits the semantic coverage and structural expres-
siveness of the generated content.
Hypergraph
Hypergraphs connect multiple vertices via hyperedges, ef-
fectively modeling complex high-order relationships among
entities and overcoming the limitation of conventional
graphs, which support only binary relations (Gao et al. 2022;
Feng et al. 2025b). These strong modeling capabilities have
led to significant progress in fields such as recommender
systems, social network analysis, and brain network mod-
eling (Ji et al. 2020; Sun et al. 2023; Han et al. 2025). How-
ever, in the RAG framework, existing research is constrained
to graph structures, primarily focusing on the pairwise rela-
tionships between entities. To model multiple entity group
semantic associations, GraphRAG generates community re-
ports through the semantic clustering of entities, while Hi-
RAG (Huang et al. 2025) incorporates hierarchical graph
knowledge via multi-level clustering. While effective in cap-
turing local relationships, these methods rely on discrete cat-
egory divisions and fail to model higher-order dependencies,
resulting in information loss.
In contrast, hypergraphs naturally connect multiple enti-
ties through hyperedges, allowing them to interact with mul-
tiple hyperedges at once. This enables the capture of higher-
order dependencies in a unified framework. It avoids the
fragmentation and loss of information typical in clustering
approaches and maximizes the retention of semantic infor-
mation during text-to-graph conversion (Feng et al. 2025a;
Luo et al. 2025). The hypergraph structure enhances seman-
tic associations both within and across documents, thereby
improving the RAG system’s ability to understand context
and ensure consistency in generated responses.
Preliminary
In this section, we provide a general expression for RAG and
graph-enhanced RAG, referring to the definitions in (Edge
et al. 2024; Guo et al. 2024).
An RAG systemMgenerally includes LLM, retriever,
and corpora, which can be defined as follows:
M= (LLM,R(q,D)).(1)
Given a queryq, the retrieverRselects relevant contexts
from the corporaD, which are then used by the LLM to
generate a response.

Figure 2: The overall framework of Cog-RAG.
For the graph-enhanced RAG, the corpus is organized into
a graph structure, where vertices represent entities and edges
represent the relations. It can be formally defined as follows:
M= (LLM,R(q,D={V,E})).(2)
The queryqretrieves relevant vertices or edges from the
graph-structured corpusD={V,E}, enabling the LLM to
respond.
Method
Overview
As illustrated in Figure 2, Cog-RAG comprises two
main components: dual-hypergraph indexing and cognitive-
inspired two-stage retrieval. We construct the dual-
hypergraph with complementary semantic granularity: the
theme hypergraph captures semantic theme associations be-
tween chunks (such as storyline, narrative outline, and sum-
mary), providing global semantic theme organization; the
entity hypergraph models fine-grained high-order relations
among entities (such as persons, concepts, and events), sup-
porting local semantic relations. In the retrieval stage, mim-
icking the human “top-down” reasoning pattern, Cog-RAG
first activates relevant themes in the theme hypergraph as
global semantic anchors. Guided by these anchors, it then
retrieves related entities and relations information from the
entity hypergraph. The final response is generated via LLMs,
utilizing theme-driven, detail-recall knowledge as evidence.Dual-Hypergraph Indexing
To more effectively model complex high-order associations
among multiple entities in corpora and avoid the informa-
tion loss by graph structure, we introduce hypergraphs for
modeling. The general formulation is defined as follows:
M= (LLM,R(q,D={V,E low,Ehigh})),(3)
where hyperedges are used to represent relations.E lowde-
notes low-order pairwise entity relations, whileE highrefers
to high-order beyond pairwise multiple entities associations.
Theme-Aware Hypergraph IndexThe theme hyper-
graph is designed to model the semantic storyline structure
of a document, establishing a narrative outline that provides
cognitive guidance for subsequent detail retrieval.
Given a corpusD, such as books, reports, or manuals,
we first segment it into a set of chunks using a fixed-length
sliding window with partial overlap to maintain semantic in-
tegrity, denoted as:
D={D 1, D2, ..., D N},(4)
whereD idenotes thei-th document chunk, serving as the
basic unit for subsequent analysis.
Then, we perform semantic parsing on each chunk us-
ing LLMs to automatically extract its latent theme and as-
sociated key entities, thereby constructing a theme hyper-
graph. Specifically, we first employ predefined theme-level

extraction promptsP exttheme,Pextkey(detailed in Appendix)
to guide the LLM in performing semantic parsing for each
chunkD iand outputting the corresponding theme. Then,
further extract the key entities related to the theme. The cal-
culation process is as follows:
Etheme =LLM(P exttheme (Di))
Vkey=LLM(P extkey(Di,Etheme ))forD i∈ D.(5)
Based on the extracted themes and entities, we can construct
the theme hypergraphG theme, denoted as:
Gtheme ={V key,Etheme},(6)
where each hyperedgeE theme represents the narrative theme
of the chunk, while the verticesV keyare the key entities.
Fine-Grained Entity Hypergraph IndexAfter con-
structing the theme hypergraph, we obtain a global thematic
structure among chunks. To further capture fine-grained
multi-entity relations, we construct an entity hypergraph
within each chunk to model high-order relations among en-
tities, supporting subsequent fine-grained retrieval.
For each chunkD i, we first extract entities (such as per-
son, event, organization, etc.) and their descriptions using
LLMs, which serve as the vertex set for the fine-grained
entity hypergraph. Based on the semantic relations among
these entities, we then construct two types of hyperedges:
low-order hyperedgesE lowcapture basic pairwise relations,
while high-order hyperedgesE highmodel more complex se-
mantic associations among multiple entities, such as co-
occurrence in events or causal links. The extraction process
is represented as follows:


V=LLM(P extentity(Di))
Elow=LLM(P extlow(Di,V))
Ehigh=LLM(P exthigh(Di,V))forD i∈ D,(7)
whereP extentity refers to the prompt designed for entity ex-
traction from the text.P extlowandP exthigh(detailed in Ap-
pendix) represent the extraction of paired and group rela-
tions from the obtained entities, respectively.
Finally, all extracted entities, along with their low-order
and high-order relations, are organized into a fine-grained
entity hypergraphG entityand stored in a hypergraph database.
Gentity ={V,E low,Ehigh}.(8)
Cognitive-Inspired Two-Stage Retrieval
Motivated by the top-down information processing pattern
observed in human memory retrieval, we design a cognitive-
inspired two-stage retrieval strategy. Specifically, it first
identify theme threads in the theme hypergraph related to the
query. These threads then serve as cues to guide the retrieval
of fine-grained information from the entity hypergraph.
For a given user queryq, following (Guo et al. 2024; Feng
et al. 2025a), we first extract two types of keywords: theme
keywords (overarching concepts or themes) and entity key-
words (specific entities or details), as follows:
Xtheme,Xentity =LLM(P keyword (q)),(9)
whereX ∗={x 1, x2, ...},P keyword is the prompt for extract-
ing keywords from the query, detailed in Appendix.Theme-Aware Hypergraph RetrievalSubsequently,
based on the extracted keywords, we perform structured
retrieval over the hypergraph database. It is worth noting
that entity keywords primarily describe concrete individual
information and are thus matched to vertices, while theme
keywords reflect abstract semantic relations among multiple
entities and are therefore used to retrieve relevant hyper-
edges. The natural combination of two types of keywords
and hypergraph structure enhances both retrieval specificity
and structural compatibility.
In the first stage of retrieval, the extracted theme keywords
are used to perform semantic matching within the theme hy-
pergraph, and selecting the top-k relevant theme hyperedges.
Erel={R(x i,Etheme )|xi∈ X theme},(10)
whereE relrepresents the relevant hyperedges retrieved from
the vector database. Then, we perform a diffusion process
over the hypergraph database to retrieve their neighboring
vertices, providing additional context awareness of the re-
trieved theme.
Vdif={N(e i,Gtheme )|ei∈ E rel},(11)
whereNdenotes the function of obtaining the correspond-
ing neighbors from the hypergraph.V difis the diffusion ver-
tices. Then, both theE relandV dif, along with the correspond-
ing textual contexts, are fed into LLMs as prior knowledge
to generate an initial theme-aware answer as follows:
Atheme =LLM(q,E rel,Vdif,Cerel,Cvdif),(12)
whereA theme denotes the output of queryqafter retrieving
fromG theme,C∗is the corresponding context.
Theme-aligned Entity Hypergraph RetrievalAfter
completing the initial theme-based retrieval, we further per-
form fine-grained information retrieval within the entity hy-
pergraph. Guided by the retrieved themes, this section sup-
plements entity-level semantic details, enabling effective
alignment between local information and global themes.
Unlike the theme retrieval stage which targets hyperedges,
this stage focuses on retrieving top-k vertices within the en-
tity hypergraph by entity keywords, thereby achieving fine-
grained semantic supplement and structured alignment.
Vrel={R(P align(xi,Atheme ),V entity)|xi∈ X entity},(13)
whereP alignis the prompt (detailed in Appendix) used for
embedding theme knowledge.V relrefers the retrieved rele-
vant entities. Then perform a hypergraph structure diffusion
as follows:
Edif={N(v i,Gentity)|vi∈ V rel}.(14)
Finally, the retrievedV rel, diffusionE dif, and their corre-
sponding contexts, integrated with the previous theme in-
formationA theme, to form a structured input for LLMs to
generate the final answerAfor queryq, thereby achieving
a comprehensive semantic generation process from theme
guidance to detailed support.
A=LLM(q,A theme,Vrel,Edif,Cvrel,Cedif).(15)

Figure 3: Test results by scoring. (a) is the comparison results on five datasets; (b) is the results of the neurology dataset on six
dimensions; (c) shows the evaluation results on different LLMs.
Experiments
Experimental Setup
DatasetsTo systematically evaluate our method across
diverse application scenarios, we adopt five datasets from
two benchmarks: Mix, CS, and Agriculture from the Ul-
traDomain benchmark (Qian et al. 2024), and Neurology
and Pathology from the MIRAGE benchmark (Xiong et al.
2024). UltraDomain covers typical RAG applications across
different domains, while MIRAGE focuses on medical ques-
tion answering and domain-specific knowledge coverage.
The statistical information is given in the Appendix.
Based on domain consistency and semantic correlation
within the texts, we categorize the datasets into three types
to enable a comprehensive analysis of the model’s adapt-
ability:Cross-domain Sparse(Mix): Fragmented passages
from unrelated domains with weak semantic coherence.
Intra-domain Sparse(CS, Agriculture): Domain-specific
documents with weak inter-passage context.Intra-domain
Dense(Neurology, Pathology): Highly structured medical
textbooks with strong semantic continuity from MIRAGE.
Additionally, we follow the data processing and query pro-
cedure of LightRAG, utilizing GPT-4o to generate complex,
document-related queries.
BaselinesWe compared our approach with the state-of-
the-art and popular RAG methods. Including text-base RAG:
NaiveRAG (Gao et al. 2023), graph-enhanced RAG ap-
proaches: GraphRAG (Edge et al. 2024), LightRAG (Guo
et al. 2024), HiRAG (Huang et al. 2025), hypergraph-
enhanced methods: Hyper-RAG (Feng et al. 2025a). The
baseline details are provided in the Appendix.
Implementation DetailsTo ensure fairness and consis-
tency for both the baseline and proposed methods, we val-
idate on five different LLMs for information extraction,
question answering, including GPT-4o-mini (Achiam et al.
2023), Qwen-Plus (Yang et al. 2025), GLM-4-Air (GLM
et al. 2024), DeepSeek-V3 (Liu et al. 2024), and LLaMa-
3.3-70B (Dubey et al. 2024). The result evaluation is default
on GPT-4o-mini, as well as the text-embedding-3-small em-bedding model for vector encoding and retrieval tasks. Un-
less otherwise specified, all reported results are based on
GPT-4o-mini.
Evaluation MetricsFollowing the recent works, we adopt
two evaluation strategies: Selection-based (Guo et al. 2024;
Huang et al. 2025) and Score-based (Wang et al. 2024; Feng
et al. 2025a), providing both relative and absolute perspec-
tives on model performance.The Selection-basedevalua-
tion uses LLMs to reports win rates of answer quality be-
tween two methods.The Score-basedevaluation employs
LLMs to score responses for different methods. Both strate-
gies assess models from six dimensions: Comprehensive-
ness, Empowerment, Relevance, Consistency, Clarity, and
Logical. We report both per-dimension and overall average
scores. Detailed evaluation descriptions are in the Appendix.
Main Results
Our primary results are presented in Table 1 and Figure 3,
and more results are provided in the Appendix. Cog-RAG
consistently outperforms all baselines across multiple di-
mensions. Additionally, we have several key insights:
1)Knowledge graphs can enhance RAG to model a
broader scope of information.Graph-enhanced methods,
represented by GraphRAG and LightRAG, demonstrate sig-
nificant advantages over the conventional NaiveRAG, pri-
marily due to the modeling of graph structures. In contrast,
NaiveRAG relies solely on vector similarity and fails to ac-
count for these structured semantic relations. Hypergraph-
enhanced approaches, such as Hyper-RAG and Cog-RAG,
offer a more comprehensive modeling of knowledge struc-
tures that extend beyond pairwise relations, demonstrating
superior potential in knowledge representation.
2)Cog-RAG outperforms the baselines across all kinds
of evaluation datasets and LLMs.ForSelection-based re-
sultsin Table 1, we can see that in cross-domain sparse
settings, both Hyper-RAG and Cog-RAG utilize hyper-
graphs to capture high-order relations, resulting in an av-
erage improvement of over 10.0% compared to graph-based
methods. In intra-domain sparse datasets, Cog-RAG outper-

Mix CS Agriculture Neurology Pathology
NaiveRAGCog-RAGNaiveRAGCog-RAGNaiveRAGCog-RAGNaiveRAGCog-RAGNaiveRAGCog-RAG
Comp. 12.0%88.0%4.0%96.0%1.0%99.0%3.0%97.0%6.0%94.0%
Empo. 10.0%90.0%3.0%97.0%2.0%98.0%1.0%99.0%4.0%96.0%
Rele. 27.0%73.0%18.0%82.0%6.0%94.0%11.0%89.0%8.0%92.0%
Cons. 10.0%90.0%4.0%96.0%1.0%99.0%2.0%98.0%4.0%96.0%
Clar. 23.0%77.0%11.0%89.0%6.0%94.0%6.0%94.0%8.0%92.0%
Logi. 11.0%89.0%5.0%95.0%1.0%99.0%1.0%99.0%5.0%95.0%
Overall 15.5%84.5%7.5%92.5%2.8%97.2%3.2%96.0%5.8%94.2%
GraphRAGCog-RAGGraphRAGCog-RAGGraphRAGCog-RAGGraphRAGCog-RAGGraphRAGCog-RAG
Comp. 40.0%60.0%36.0%64.0%32.0%68.0%34.0%66.0%32.0%68.0%
Empo. 36.0%64.0%35.0%65.0%26.0%74.0%27.0%73.0%23.0%77.0%
Rele. 45.0%55.0%39.0%61.0%35.0%65.0%37.0%63.0%31.0%69.0%
Cons. 40.0%60.0%35.0%65.0%29.0%71.0%31.0%69.0%31.0%69.0%
Clar. 46.0%54.0%36.0%64.0%38.0%62.0%36.0%64.0%30.0%70.0%
Logi. 39.0%61.0%37.0%63.0%27.0%73.0%33.0%67.0%29.0%71.0%
Overall 41.0%59.0%36.3%63.7%31.2%68.8%33.0%67.0%29.5%70.5%
LightRAGCog-RAGLightRAGCog-RAGLightRAGCog-RAGLightRAGCog-RAGLightRAGCog-RAG
Comp. 38.0%62.0%30.0%70.0%23.0%77.0%28.0%72.0%30.0%70.0%
Empo. 30.0%70.0%26.0%74.0%20.0%80.0%22.0%78.0%25.0%75.0%
Rele. 36.0%64.0%27.0%73.0%25.0%75.0%28.0%72.0%32.0%68.0%
Cons. 34.0%66.0%29.0%71.0%21.0%79.0%25.0%75.0%27.0%73.0%
Clar. 38.0%62.0%24.0%76.0%22.0%78.0%26.0%74.0%26.0%74.0%
Logi. 35.0%65.0%29.0%71.0%23.0%77.0%26.0%74.0%26.0%74.0%
Overall 35.2%64.8%27.5%72.5%22.3%77.7%25.8%74.2%27.7%72.3%
HiRAGCog-RAGHiRAGCog-RAGHiRAGCog-RAGHiRAGCog-RAGHiRAGCog-RAG
Comp. 44.0%56.0%40.0%60.0%41.0%59.0%35.0%65.0%40.0%60.0%
Empo. 39.0%61.0%36.0%64.0%36.0%64.0%31.0%69.0%37.0%63.0%
Rele. 45.0%55.0%47.0%53.0%44.0%56.0%35.0%65.0%41.0%59.0%
Cons. 39.0%61.0%40.0%60.0%37.0%63.0%32.0%68.0%37.0%63.0%
Clar. 45.0%54.0%50.0%50.0%44.0%56.0%31.0%69.0%40.0%60.0%
Logi. 40.0%60.0%40.0%60.0%38.0%62.0%31.0%69.0%36.0%64.0%
Overall 42.0%58.0%42.2%57.8%40.0%60.0%32.5%67.5%38.5%61.5%
Hyper-RAGCog-RAGHyper-RAGCog-RAGHyper-RAGCog-RAGHyper-RAGCog-RAGHyper-RAGCog-RAG
Comp. 43.0%57.0%45.0%55.0%49.0%51.0%40.0%60.0%42.0%58.0%
Empo. 42.0%58.0%43.0%57.0%40.0%60.0%37.0%63.0%37.0%63.0%
Rele.53.0%47.0% 47.0%53.0%45.0%55.0%46.0%54.0%37.0%63.0%
Cons. 43.0%57.0%44.0%56.0%44.0%56.0%38.0%62.0%36.0%64.0%
Clar.56.0% 44.0% 48.0%52.0%42.0%58.0%41.0%59.0%32.0%68.0%
Logi. 44.0%56.0%46.0%54.0%43.0%57.0%35.0%65.0%37.0%63.0%
Overall 46.8%53.2%45.5%54.5%43.8%56.2%39.5%60.5%36.8%63.2%
Table 1: Average win rates of six evaluation metrics across five datasets. The comparison is made between baselines and Cog-
RAG. Among them, we refer to the six metrics as Comp. (Comprehensiveness), Empo. (Empowerment), Rele. (Relevance),
Cons. (Consistency), Clar. (Clarity), and Logi. (Logical).
forms HiRAG by 15.6% and 20.0%, benefiting from multi-
hyperedge propagation that uncovers latent themes and en-
tity relations. In intra-domain dense medical corpora, Cog-
RAG achieves the most significant gains. Through dual hy-
pergraph modeling and cognitive-inspired retrieval, enhanc-
ing the alignment and aggregation of theme and fine-grained
details. Compared to Hyper-RAG, it improves by 21.0% and
26.4%, respectively. ForScore-based results, Figure 3 ob-
jectively presents the evaluation results across six dimen-
sions on five LLMs. The results demonstrate that Cog-RAG
achieves consistent and significant improvements over base-line methods in all dimensions. Moreover, when applying
different LLMs for indexing and answering, it still exhibits
clear advantages, highlighting its structural effectiveness.
3)Dual-hypergraph alignment enhances knowledge
representation and semantic consistency.Inspired by hu-
man top-down cognitive pathways, Cog-RAG utilizes a dual
hypergraph structure to align macro to micro knowledge.
Specifically, forSelection-based resultsin Intra-domain
Dense scenario, Cog-RAG improves by 35.0% and 23.0%
compared to the entity-level hierarchical method HiRAG.
ForScore-based results, Cog-RAG outperforms HiRAG and

ModelsMix CS Neurology
(Overall) (Overall) (Overall)
COG-RAG 85.39 87.07 86.55
w./o. Entity Hypergraph76.58 84.58 84.49
w./o. Theme Hypergraph84.82 85.88 85.41
w./o. Two-Stage Retrieval84.88 86.41 86.18
Table 2: Ablation study on different datasets by scoring,
where w./o. indicates without the part of the method.
Hyper-RAG by 1.37 and 1.20 on neurology datasets, signif-
icantly enhancing the model’s ability to handle knowledge-
intensive domains and ensuring semantic consistency.
Ablation Study
This section conducts ablation studies to evaluate the con-
tribution of each core component in Cog-RAG: the theme,
entity hypergraph, and two-stage retrieval strategy. Ta-
ble 2 shows the results by Scoring-based evaluation, and
Selection-based results can be found in the Appendix. The
results from three types of representative datasets are sum-
marized below.
1)Effectiveness of the Entity Hypergraph.Removing
the entity hypergraph leads to a significant decrease in per-
formance on all three types of datasets, especially on the
Mix dataset. This indicates its critical role in capturing fine-
grained semantic relations within chunks. This effect is con-
sistently observed across domains, confirming that intra-
chunk entity-level modeling can enhance the representation
of local knowledge.
2)Effectiveness of the Theme Hypergraph.Excluding
the theme hypergraph causes a moderate decrease (drop 1.19
on CS and 1.14 on Neurology), highlighting its role in mod-
eling global theme structures across chunks. The benefit is
particularly noticeable in intra-domain tasks, where main-
taining coherent theme alignment helps with cross-chunk
reasoning and retrieval. However, on the Mix dataset, using
only the theme hypergraph leads to performance degrada-
tion (from 85.39 to 76.58), indicating that in cross-domain
sparse and weakly structured scenarios, theme relations may
introduce noise that interferes with retrieval and answering.
3)Effectiveness of the Two-Stage Retrieval.Bypassing
this component (by directly concatenating information from
both the theme and entity hypergraphs and inputting it into
LLMs) leads to consistent performance drops. This high-
lights the importance of the two-stage retrieval, especially in
knowledge-intensive scenarios where global semantic guid-
ance followed by entity-level refinement enables more accu-
rate and coherent retrieval.
Hypergraph Visualization
In the Neurology dataset, Figure 4 visualized the relations
of Sleep Apnea in the entity hypergraph. It illustrates the
complex relations between Sleep Apnea and multiple related
entities such as Chronic Lung Disease, Headache, and Res-
piratory Centers. It captures not only pairwise relations but
Figure 4: Entity Hypergraph Visualization.
also reveals multi-entity dependencies beyond pairs. As ob-
served, the complex hypergraph among Hypertension, Sleep
Apnea, Kyphoscoliosis, and Muscular Dystrophy illustrates
various health risks and respiratory challenges connected to
sleep quality and disorders, affecting overall wellness.
Why is Cog-RAG Effective?
Theme-Aligned vs. Graph / Hypergraph Index
Graph and hypergraph-enhanced RAG mainly focus on
modeling local entity-level relations within document
chunks, making them less effective for tasks that require
global semantic reasoning. In contrast, Cog-RAG introduces
a dual-hypergraph structure that supports alignment from
global themes to fine-grained entities, leading to improved
contextual grounding and response consistency. Notably, our
analysis reveals that the theme hypergraph is particularly
beneficial in structured, domain-specific settings, while it
may introduce noise in loosely structured, open-domain sce-
narios. This suggests further opportunities for dynamic fil-
tering and graph construction.
Cognitive-Inspired vs. Conventional Retrieval
Conventional RAG systems rely on single-stage retrieval,
which merges all retrieved content into LLMs. This de-
sign often leads to incomplete or noisy evidence aggrega-
tion for complex knowledge-intensive tasks. The cognitive-
inspired two-stage retrieval strategy enables top-down se-
mantic alignment and aggregation, providing more accurate
knowledge support and reducing redundant information.
Conclusion
Inspired by human cognitive pathways, this paper intro-
duces Cog-RAG, which enhances LLM responses by inte-
grating dual-hypergraph structures and a cognitive-inspired
two-stage retrieval mechanism. Cog-RAG enables hierarchi-
cal knowledge modeling and semantic alignment at both
macro-thematic and micro-entity levels, addressing issues of
information loss and semantic gaps inherent in graph-based
methods. Experimental results show that Cog-RAG signif-
icantly outperforms state-of-the-art methods across various
types of datasets on knowledge-intensive tasks.

Acknowledgement
This work was supported by the National Natural Sci-
ence Foundation of China under Grant Nos. 62088102 and
U24A20252, the Key Research and Development Program
of Shaanxi Province of China under Grant Nos. 2024PT-
ZCK-66 and 2024CY2-GJHX-48.
References
Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya,
I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Alt-
man, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report.
arXiv:2303.08774.
Asai, A.; Wu, Z.; Wang, Y .; Sil, A.; and Hajishirzi, H. 2023.
Self-rag: Learning to retrieve, generate, and critique through
self-reflection. InThe Twelfth International Conference on
Learning Representations.
Ayala, O.; and Bechard, P. 2024. Reducing hallucination
in structured outputs via Retrieval-Augmented Generation.
Proceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 6: Industry Track),
228–238.
Chen, B.; Guo, Z.; Yang, Z.; Chen, Y .; Chen, J.; Liu,
Z.; Shi, C.; and Yang, C. 2025. Pathrag: Pruning graph-
based retrieval augmented generation with relational paths.
arXiv:2502.14902.
Cheng, Y .; Zhao, Y .; Zhu, J.; Liu, Y .; Sun, X.; and Li, X.
2025. Human Cognition Inspired RAG with Knowledge
Graph for Complex Problem Solving. arXiv:2503.06567.
Dong, G.; Song, X.; Zhu, Y .; Qiao, R.; Dou, Z.; and Wen,
J.-R. 2025. Toward verifiable instruction-following align-
ment for retrieval augmented generation. InProceedings of
the AAAI Conference on Artificial Intelligence, volume 39,
23796–23804. Philadelphia, Pennsylvania, USA.
Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.;
Letman, A.; Mathur, A.; Schelten, A.; Yang, A.; Fan, A.;
et al. 2024. The llama 3 herd of models. arXiv:2407.21783.
Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.; Mody,
A.; Truitt, S.; Metropolitansky, D.; Ness, R. O.; and Lar-
son, J. 2024. From local to global: A graph rag approach to
query-focused summarization. arXiv:2404.16130.
Fan, W.; Ding, Y .; Ning, L.; Wang, S.; Li, H.; Yin, D.; Chua,
T.-S.; and Li, Q. 2024. A survey on rag meeting llms: To-
wards retrieval-augmented large language models. InPro-
ceedings of the 30th ACM SIGKDD conference on knowl-
edge discovery and data mining, 6491–6501. New York, NY ,
USA.
Feng, Y .; Hu, H.; Hou, X.; Liu, S.; Ying, S.; Du, S.; Hu,
H.; and Gao, Y . 2025a. Hyper-RAG: Combating LLM Hal-
lucinations using Hypergraph-Driven Retrieval-Augmented
Generation. arXiv:2504.08758.
Feng, Y .; Yang, C.; Hou, X.; Du, S.; Ying, S.; Wu, Z.; and
Gao, Y . 2025b. Beyond Graphs: Can Large Language Mod-
els Comprehend Hypergraphs? InThe Thirteenth Inter-
national Conference on Learning Representations, 42468–
42495. Singapore.Gao, Y .; Feng, Y .; Ji, S.; and Ji, R. 2022. HGNN+: General
hypergraph neural networks.IEEE Transactions on Pattern
Analysis and Machine Intelligence, 45(3): 3181–3199.
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai,
Y .; Sun, J.; Wang, H.; and Wang, H. 2023. Retrieval-
augmented generation for large language models: A survey.
arXiv:2312.10997.
GLM, T.; Zeng, A.; Xu, B.; Wang, B.; Zhang, C.; Yin, D.;
Zhang, D.; Rojas, D.; Feng, G.; Zhao, H.; et al. 2024. Chat-
glm: A family of large language models from glm-130b to
glm-4 all tools. arXiv:2406.12793.
Guo, Z.; Xia, L.; Yu, Y .; Ao, T.; and Huang, C. 2024.
Lightrag: Simple and fast retrieval-augmented generation.
arXiv:2410.05779.
Guti´errez, B. J.; Shu, Y .; Gu, Y .; Yasunaga, M.; and Su, Y .
2024. HippoRAG: Neurobiologically Inspired Long-Term
Memory for Large Language Models. InAdvances in Neural
Information Processing Systems, 59532–59569. Red Hook,
NY , USA.
Han, X.; Xue, R.; Feng, J.; Feng, Y .; Du, S.; Shi, J.; and
Gao, Y . 2025. Hypergraph foundation model for brain dis-
ease diagnosis.IEEE Transactions on Neural Networks and
Learning Systems, 1–15.
Huang, H.; Huang, Y .; Yang, J.; Pan, Z.; Chen, Y .; Ma, K.;
Chen, H.; and Cheng, J. 2025. Retrieval-Augmented Gener-
ation with Hierarchical Knowledge. arXiv:2503.10150.
Ji, S.; Feng, Y .; Ji, R.; Zhao, X.; Tang, W.; and Gao, Y . 2020.
Dual channel hypergraph collaborative filtering. InProceed-
ings of the 26th ACM SIGKDD international conference on
knowledge discovery & data mining, 2020–2029. New York,
NY , USA.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks.Advances in neural infor-
mation processing systems, 33: 9459–9474.
Li, R.; and Du, X. 2023. Leveraging structured informa-
tion for explainable multi-hop question answering and rea-
soning. InFindings of the Association for Computational
Linguistics: EMNLP 2023, 6779–6789. Singapore.
Li, Z.; Chen, X.; Yu, H.; Lin, H.; Lu, Y .; Tang, Q.; Huang,
F.; Han, X.; Sun, L.; and Li, Y . 2024. Structrag: Boosting
knowledge intensive reasoning of llms via inference-time
hybrid information structurization. arXiv:2410.08815.
Liu, A.; Feng, B.; Xue, B.; Wang, B.; Wu, B.; Lu, C.; Zhao,
C.; Deng, C.; Zhang, C.; Ruan, C.; et al. 2024. Deepseek-v3
technical report. arXiv:2412.19437.
Luo, H.; Chen, G.; Zheng, Y .; Wu, X.; Guo, Y .; Lin,
Q.; Feng, Y .; Kuang, Z.; Song, M.; Zhu, Y .; et al.
2025. HyperGraphRAG: Retrieval-Augmented Genera-
tion via Hypergraph-Structured Knowledge Representation.
arXiv:2503.21322.
Peng, B.; Zhu, Y .; Liu, Y .; Bo, X.; Shi, H.; Hong, C.; Zhang,
Y .; and Tang, S. 2024. Graph retrieval-augmented genera-
tion: A survey. arXiv:2408.08921.

Qian, H.; Zhang, P.; Liu, Z.; Mao, K.; and Dou, Z.
2024. Memorag: Moving towards next-gen rag via memory-
inspired knowledge discovery. arXiv:2409.05591.
Sarmah, B.; Mehta, D.; Hall, B.; Rao, R.; Patel, S.; and
Pasquali, S. 2024. Hybridrag: Integrating knowledge graphs
and vector retrieval augmented generation for efficient in-
formation extraction. InProceedings of the 5th ACM Inter-
national Conference on AI in Finance, 608–616. New York,
NY , USA.
Sun, X.; Cheng, H.; Liu, B.; Li, J.; Chen, H.; Xu, G.; and
Yin, H. 2023. Self-supervised hypergraph representation
learning for sociological analysis.IEEE Transactions on
Knowledge and Data Engineering, 35(11): 11860–11871.
Wang, M.; Chen, L.; Fu, C.; Liao, S.; Zhang, X.; Wu, B.;
Yu, H.; Xu, N.; Zhang, L.; Luo, R.; et al. 2024. Leave no
document behind: Benchmarking long-context llms with ex-
tended multi-doc qa. InProceedings of the 2024 Confer-
ence on Empirical Methods in Natural Language Process-
ing, 5627–5646. Miami, Florida, USA.
Wang, S.; Fang, Y .; Zhou, Y .; Liu, X.; and Ma, Y .
2025. ArchRAG: Attributed Community-based Hierarchi-
cal Retrieval-Augmented Generation. arXiv:2502.09891.
Xia, Y .; Zhou, J.; Shi, Z.; Chen, J.; and Huang, H. 2025.
Improving retrieval augmented language model with self-
reasoning. InProceedings of the AAAI conference on artifi-
cial intelligence, volume 39, 25534–25542.
Xiong, G.; Jin, Q.; Lu, Z.; and Zhang, A. 2024. Benchmark-
ing retrieval-augmented generation for medicine. InFind-
ings of the Association for Computational Linguistics ACL
2024, 6233–6251. Bangkok, Thailand.
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3
technical report. arXiv:2505.09388.
Yang, X.; Sun, K.; Xin, H.; Sun, Y .; Bhalla, N.; Chen, X.;
Choudhary, S.; Gui, R.; Jiang, Z.; Jiang, Z.; et al. 2024.
Crag-comprehensive rag benchmark. InAdvances in Neural
Information Processing Systems, volume 37, 10470–10490.
Red Hook, NY , USA.
Zhang, L.; Yu, Y .; Wang, K.; and Zhang, C. 2024. ARL2:
Aligning Retrievers with Black-box Large Language Mod-
els via Self-guided Adaptive Relevance Labeling. InPro-
ceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics, 3708–3719. Bangkok, Thailand.
Zhang, Q.; Chen, S.; Bei, Y .; Yuan, Z.; Zhou, H.; Hong, Z.;
Dong, J.; Chen, H.; Chang, Y .; and Huang, X. 2025. A sur-
vey of graph retrieval-augmented generation for customized
large language models. arXiv:2501.13958.

APPENDIX
A. Experimental Datasets
Statistics Mix CS Agriculture Neurology Pathology
Total Documents 61 10 12 1 1
Total Chunks 560 1992 1813 1790 824
Total Tokens 615,355 2,190,803 1,993,515 1,968,716 905,760
Table 3: Statistical information of the datasets
Table 1 presents the statistical information of the five
datasets.
B. Details of Baselines
We compared our approach with the state-of-the-art and
popular RAG methods. The specific description of the
method is as follows:
NaiveRAG: The baseline of the standard RAG systems,
which segments texts into chunks and stores them as embed-
dings in a vector database. During retrieval, relevant chunks
are directly matched via vector similarity.
GraphRAG: A standard graph-enhanced RAG method
that employs LLMs to extract entities and relations from
texts as nodes and edges of the graph. Entities are fur-
ther clustered to generate community reports, which are tra-
versed during retrieval to obtain global information.
LightRAG: A graph-enhanced RAG method that inte-
grates graph structures with vector-based representations. It
employs a dual-level retrieval strategy to retrieve informa-
tion from both nodes and edges within the graph knowledge.
HiRAG: A graph-enhanced RAG approach that builds a
hierarchical graph via multi-level semantic clustering, en-
abling hierarchical indexing and retrieval of text knowledge.
Hyper-RAG: A standard Hypergraph-enhanced RAG
method that uses hyperedges to represent both paired low-
order relations and beyond-paired high-order relations.
C. Details of Evaluation Metrics
We evaluated the proposed method and baseline from six
dimensions, including Comprehensiveness, Empowerment,
Relevance, Consistency, Clarity, and Logical coherence.
For Selection-based evaluation, we report win rates to
conduct a qualitative comparison. We also alternated the an-
swer order of each pair of methods in the prompts and cal-
culated the average result for a fair comparison. The metrics
details are described below:
Comprehensiveness: How much detail does the answer
provide to cover all aspects and details of the question?
Empowerment: How well does the answer help the reader
understand and make informed judgments about the topic?
Relevance: How precisely does the answer address the core
aspects of the question without including unnecessary infor-
mation?Consistency: How well does the system integrate
and synthesize information from multiple sources into a log-
ically flowing response?Clarity: How well does the systemprovide complete information while avoiding unnecessary
verbosity and redundancy?Logical: How well does the sys-
tem maintain consistent logical arguments without contra-
dicting itself across the response?
For Score-based evaluation, we use LLMs to score re-
sponses for quantitative assessment, with the following spe-
cific indicators:
Comprehensiveness (0-100): Measure whether the an-
swer comprehensively covers all key aspects of the ques-
tion and whether there are omissions.Empowerment (0-
100): Measure the credibility of the answer and whether it
convinces the reader that it is correct. High confidence an-
swers often cite authoritative sources or provide sufficient
evidence.Relevance (0-100): Measure whether the content
of the answer is closely related to the question, and whether
it stays focused on the topic without digression.Consistency
(0-100): Measure whether the answer is logically organized,
flows smoothly, and whether the parts of the answer are well
connected and mutually supportive.Clarity (0-100): Mea-
sure whether the answer is expressed in a clear, unambigu-
ous, and easily understandable manner, using appropriate
language and definitions.Logical (0-100): Measure whether
the answer is coherent, clear, and easy to understand.
For each evaluation dimension, we define five discrete
rating levels, each associated with clear scoring criteria to
ensure consistency and transparency. As an illustration, the
Comprehensiveness dimension is rated as follows:Level 1
(0-20): The answer is extremely one-sided, leaving out key
parts or important aspects of the question.Level 2 (20-40):
The answer has some content but misses many important
aspects and is not comprehensive enough.Level 3 (40-60):
The answer is more comprehensive, covering the main as-
pects of the question, but there are still some omissions.
Level 4 (60-80): The answer is comprehensive, covering
most aspects of the question with few omissions.Level 5
(80-100): The answer is extremely comprehensive, covering
all aspects of the question with no omissions, enabling the
reader to gain a complete understanding.
D. Additional Experiment Results
D.1 Comparison Experiment Results
Figure 1 shows the results of comparison experiments under
different evaluation dimensions on the Mix, CS, Agriculture,
and Pathology datasets.
D.2 Ablation Experiment Results
Table 2 presents the results of the selection-based evaluation
on three types of representative datasets, including Mix, CS,
and Neurology. The results also show the effectiveness of
each component in Cog-RAG.
E. Retrieval Efficiency Analysis
We conduct a comparative analysis of retrieval efficiency.
Figure 2 illustrates the trade-off between retrieval time and
final answer score across different methods. Notably, Cog-
RAG achieves the highest overall score while maintaining
low retrieval overhead, highlighting its superior balance be-
tween performance and efficiency.

Figure 5: Comparison results on different metrics.
CS Mix Neurology
w./o. Entity
HypergraphCog-RAGw./o. Entity
HypergraphCog-RAGw./o. Entity
HypergraphCog-RAG
Comp. 24.0%76.0%34.0%66.0%30.0%70.0%
Empo. 20.0%80.0%41.0%59.0%23.0%77.0%
Rele. 29.0%71.0%31.0%69.0%30.0%70.0%
Cons. 24.0%76.0%41.0%59.0%26.0%74.0%
Clar. 25.0%75.0%25.0%75.0%30.0%70.0%
Logi. 24.0%76.0%44.0%56.0%28.0%72.0%
Overall 24.0%76.0%36.0%64.0%28.0%72.0%
w./o. Theme
HypergraphCog-RAGw./o. Theme
HypergraphCog-RAGw./o. Theme
HypergraphCog-RAG
Comp. 46.0%54.0%41.0%59.0%43.0%57.0%
Empo. 45.0%55.0%39.0%61.0%39.0%61.0%
Rele. 47.0%53.0%39.0%61.0%42.0%58.0%
Cons. 45.0%55.0%41.0%59.0%41.0%59.0%
Clar.57.0%43.0% 39.0%61.0%41.0%59.0%
Logi. 48.0%52.0%40.0%60.0%41.0%59.0%
Overall 48.0%52.0%40.0%60.0%41.0%59.0%
w./o. Two-Stage
RetrievalCog-RAGw./o. Two-Stage
RetrievalCog-RAGw./o. Two-Stage
RetrievalCog-RAG
Comp. 44.0%56.0%46.0%54.0%43.0%57.0%
Empo 40.0%60.0%44.0%56.0%40.0%60.0%
Rele.51.0%49.0% 45.0%55.0%39.0%61.0%
Cons. 39.0%61.0%47.0%53.0%44.0%56.0%
Clar. 50.0%50.0%49.0%51.0%49.0%51.0%
Logi. 38.0%62.0%47.0%53.0%45.0%55.0%
Overall 44.0%56.0%46.0%54.0%43.0%57.0%
Table 4: Average win rates on three datasets. The comparison is made between ablation and Cog-RAG.

0.0 0.5 1.0 1.5 2.08384858687
NaiveRAGGraphRAG
LightRAGHiRAG Hyper-RAGCog-RAGScore
Time for Retrieval (s)Figure 6: Comparison results on different metrics.
F. Prompt Templates in Cog-RAG
F.1 Extracting Themes, Key Entities
Extracting Themes
Formulation:P exttheme
Prompt:Summarize the primary theme of the text
document. This summary should capture the essence
of the document’s core conflict, main idea, or narra-
tive arc. Ensure that the summary highlights key mo-
ments, changes, or shifts in the document, extract the
following information:
- theme description: A sentence that describes the
primary theme of the text document, reflecting the
main conflict, resolution, or key message.
Extracting Key Entities
Formulation:P exttheme
Prompt:From the theme identified inP extkey, iden-
tify all key entities in each text document.
For each identified key entity, extract the following in-
formation:
- key entity name: Name of the key entity, use same
language as input text. If English, capitalized the
name.
- key entity type: Type of the key entity, such as per-
son, concept, object, event, emotion, symbol.
- key entity description: Comprehensive description
of the entity’s attributes and activities.
- key score: A score from 0 to 100 indicating the im-
portance of the entity in the text.F.2 Extracting Entities, Relations and Keywords
Extracting Entities
Formulation:P extentity
Prompt:Identify all entities. For each identi-
fied entity, extract the following information:
- entity name: Name of the entity, use same language
as input text. If English, capitalized the name.
- entity type: One of the following types: organiza-
tion, person, geo, event, role, concept.
- entity description: Comprehensive description of
the entity’s attributes and activities.
- additional properties: Other attributes possibly
associated with the entity, like time, space, emotion,
motivation, etc.
Extracting Low-order Relations
Formulation:P extlow
Prompt:From the entities identified inP extentity,
identify all pairs of (source entity, target entity) that
are *clearly related* to each other.
For each pair of related entities, extract the following
information:
- entities pair: The name of source entity and target
entity, as identified inP extentity.
- low order relationship description: Explanation
as to why you think the source entity and the target
entity are related to each other.
- low order relationship keywords: Keywords that
summarize the overarching nature of the relation-
ship, focusing on concepts or themes rather than
specific details.
- low order relationship strength: A numerical score
indicating the strength of the relationship between
the entities.
Extracting Keywords from Query
Formulation:P keywords
Prompt:You are a helpful assistant tasked
with identifying both theme-level and entity-level
keywords in the user’s query.
—Goal—
Given the query and its related theme, please refer
to the theme, list both theme-level and entity-level
keywords. theme-level keywords focus on overarch-
ing concepts or themes, while entity-level keywords
focus on specific entities, details, or concrete terms.

Extracting High-order Relations
Formulation:P exthigh
Prompt:For the entities identified inP extentity, based on the entity pair relationships inP extlow, find connec-
tions or commonalities among multiple entities and construct high-order associated entity set as much as possible.
Extract the following information from all related entities, entity pairs:
- entities set: The collection of names for elements in high-order associated entity set. -
high order relationship description: Use the relationships among the entities in the set to create a detailed,
smooth, and comprehensive description that covers all entities in the set, without leaving out any relevant information.
- high order relationship generalization: Summarize the content of the entity set as concisely as possible.
- high order relationship keywords: Keywords that summarize the overarching nature of the high-order association,
focusing on concepts or themes rather than specific details.
- high order relationship strength: A numerical score indicating the strength of the association among the entities in
the set.
F.3 Theme Alignment to Entity
Theme Alignment to Entity
Formulation:P algin
Prompt:Through the existing analysis, we can know the potential theme answer and entity-level keywords are
Atheme andX extentity. Please refer to the theme answer and entity-level keywords, combined with your own analysis, to
select useful and relevant information to help you answer accurately.
F.4 Evaluation Metrics
Selection-based Evaluation
Formulation:P eval scoring (q,A 1,A2)
qdenotes user query,A 1andA 2denotes the response from two approaches.
Prompt:You will evaluate two answers to the same question based on six criteria: Comprehensiveness, Empow-
erment, Relevance, Consistency, Clarity, and Logical.
—Goal—
You will evaluate two answers to the same question by using the relevant documents based on six criteria: Comprehen-
siveness, Empowerment, Relevance, Consistency, Clarity, and Logical.
-Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the ques-
tion?
-Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?
...,
-Logical: How well does the system maintain consistent logical arguments without contradicting itself across the
response?
For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an
overall winner based on these six categories.
Here are the question:q
Here are the two answers:
Answer 1:A 1;
Answer 2:A 2
Evaluate both answers using the six criteria listed above and provide detailed explanations for each criterion.

Scoring-based Evaluation
Formulation:P eval scoring (q,A,T o)
qdenotes user query,Adenotes LLM response,T odenotes the original text chunk that generated the question.
Prompt:You are an expert tasked with evaluating answers to the questions by using the relevant documents
based on five criteria: Comprehensiveness, Diversity, Empowerment, Logical, and Readability.
—Goal—
You will evaluate tht answers to the questions by using the relevant documents based on on six criteria: Comprehensive-
ness, Empowerment, Relevance, Consistency, Clarity, and Logical.
-Comprehensiveness-
Measure whether the answer comprehensively covers all key aspects of the question and whether there are omissions.
Level|score range|description
Level 1|0-20|The answer is extremely one-sided, leaving out key parts or important aspects of the question.
Level 2|20-40|The answer has some content, but it misses many important aspects of the question and is not
comprehensive enough.
Level 3|40-60|The answer is more comprehensive, covering the main aspects of the question, but there are still some
omissions.
Level 4|60-80|The answer is comprehensive, covering most aspects of the question, with few omissions.
Level 5|80-100|The answer is extremely comprehensive, covering all aspects of the question with no omissions,
enabling the reader to gain a complete understanding.
...,
For each indicator, please give the problem a corresponding Level based on the description of the indicator, and then
give a score according to the score range of the level.
Here are the question:q
Here are the relevant document:T o
Here are the answer:A
Evaluate all the answers using the six criteria listed above, for each criterion, provide a summary description,
give a Level based on the description of the indicator, and then give a score based on the score range of the level.