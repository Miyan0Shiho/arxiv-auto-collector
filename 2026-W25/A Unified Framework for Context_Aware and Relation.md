# A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation

**Authors**: Haoyang Zhong, Yifei Sun, Antong Zhang, Chunping Wang, Lei Chen, Yang Yang

**Published**: 2026-06-16 15:44:10

**PDF URL**: [https://arxiv.org/pdf/2606.18075v1](https://arxiv.org/pdf/2606.18075v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a paradigm for enhancing large language models (LLMs) with external knowledge, yet existing graph-based methods face a fundamental limitation: entity-centric and chunk-centric approaches operate on representations anchored to original text without true knowledge fusion. While entity-centric methods connect logically related content and chunk-centric methods preserve context, both retrieve information separately through similarity search, missing emergent understanding from their synthesis. In this paper, we propose HyGRAG, a hierarchical graph RAG framework that transcends source documents by addressing three core challenges: constructing summaries that genuinely integrate contextual and relational information, leveraging these synthesized representations to access emergent knowledge during retrieval, and efficiently updating hierarchical structures for dynamic corpora. Specifically, we design hierarchical index structures over hybrid graphs with both chunk and entity nodes, then iteratively cluster them and generate LLM-based summaries. Then, we design context and relation-aware retrieval that searches across all abstraction levels while expanding through community membership. Moreover, we enable dynamic knowledge update through attachment-based algorithms with only local re-summarization. Experimental results show that HyGRAG improves the average accuracy of multi-hop reasoning tasks by 9.7%, while maintaining reasonable efficiency.

## Full Text


<!-- PDF content starts -->

A Unified Framework for Context-Aware and Relation-Aware
Graph Retrieval-Augmented Generation
Haoyang Zhong∗
Zhejiang University
Hangzhou, China
Nanyang Technological University
Singapore, Singapore
haoyang011@e.ntu.edu.sgYifei Sun†
Zhejiang University
Hangzhou, China
yifeisun@zju.edu.cnAntong Zhang
Zhejiang University
Hangzhou, China
antongzhang@zju.edu.cn
Chunping Wang
Finvolution Group
Shang Hai, China
wangchunping02@xinye.comLei Chen
Finvolution Group
Shang Hai, China
chenlei04@xinye.comYang Yang
Zhejiang University
Hangzhou, China
yangya@zju.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) has emerged as a para-
digm for enhancing large language models (LLMs) with external
knowledge, yet existing graph-based methods face a fundamental
limitation: entity-centric and chunk-centric approaches operate on
representations anchored to original text without true knowledge
fusion. While entity-centric methods connect logically related con-
tent and chunk-centric methods preserve context, both retrieve
information separately through similarity search, missing emer-
gent understanding from their synthesis. In this paper, we propose
HyGRAG, a hierarchical graph RAG framework that transcends
source documents by addressing three core challenges: construct-
ing summaries that genuinely integrate contextual and relational
information, leveraging these synthesized representations to ac-
cess emergent knowledge during retrieval, and efficiently updating
hierarchical structures for dynamic corpora. Specifically, we de-
sign hierarchical index structures over hybrid graphs with both
chunk and entity nodes, then iteratively cluster them and generate
LLM-based summaries. Then, we design context and relation-aware
retrieval that searches across all abstraction levels while expanding
through community membership. Moreover, we enable dynamic
knowledge update through attachment-based algorithms with only
local re-summarization. Experimental results show thatHyGRAG
improves the average accuracy of multi-hop reasoning tasks by
9.7%, while maintaining reasonable efficiency.1
CCS Concepts
•Networks→Online social networks;•Computing method-
ologies→Knowledge representation and reasoning.
∗Work done while visiting Zhejiang University.
†Corresponding author.
1Our codes are available at https://github.com/zjunet/HyGRAG.
This work is licensed under a Creative Commons Attribution 4.0 International License.
WWW ’26, Dubai, United Arab Emirates
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2307-0/2026/04
https://doi.org/10.1145/3774904.3792720
Relation-awareContext-aware
🧐Query: After ..., was there agreement between Polygon and Engadgeton the immediate availability of the Steam Deck OLED units from Valve?
😊Answer: Yes
🧐Query: What is the name of the castle in the city Jan Klapáčwas born in?
😊Answer: Prague Castle
🔗What was Jan Klapáč‘s city of birth? --> Prague
🔗What is the name of the castle in Prague? --> Prague Castle
✖What was Casimir Pulaski‘s city of birth? --> Warsaw
🔗Polygon published a report on Steam Deck hardware updates at 18:00:00 on November 9, 2023.
🔗Engadget published a review of the Steam Deck OLED at 18:00:38 on November 9, 2023.
✖Report published by Polygon on 2023-10-07 under the category 'entertainment'.
Figure 1: Illustration of method performance using Qwen3-
8B across relation-aware (MuSiQue) and context-aware
(MultiHop-RAG) datasets, including representative cases.
Keywords
Graph RAG; context-aware; relation-aware.
ACM Reference Format:
Haoyang Zhong, Yifei Sun, Antong Zhang, Chunping Wang, Lei Chen,
and Yang Yang. 2026. A Unified Framework for Context-Aware and Relation-
Aware Graph Retrieval-Augmented Generation. InProceedings of the ACM
Web Conference 2026 (WWW ’26), April 13–17, 2026, Dubai, United Arab Emi-
rates.ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3774904.
3792720
1 Introduction
RAG systems have been developed to enhance LLMs by integrating
external knowledge sources [ 5,17]. This integration allows LLMs to
access up-to-date information and domain-specific knowledge, ad-
dressing the inherent limitations of parametric memory [ 4]. RecentarXiv:2606.18075v1  [cs.AI]  16 Jun 2026

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Haoyang Zhong et al.
advances in RAG have moved beyond simple document retrieval
to incorporate graph structures that explicitly model relationships
between concepts, entities, and documents [ 10,30]. These graph
RAG systems leverage the rich structural information in knowledge
graphs to improve both retrieval accuracy and reasoning capabili-
ties, making them particularly effective for complex queries that
require understanding interconnected information [8, 18].
Current Graph RAG methods have explored two distinct di-
rections to harness graph structures for retrieval augmentation.
Entity-centric approaches such as GraphRAG [ 3], HippoRAG [ 8],
and HiRAG [ 11] build knowledge graphs where nodes represent
entities extracted from text and edges encode their semantic rela-
tionships. These methods excel at multi-hop reasoning by enabling
traversal along relational paths—for instance, answering "Which
company acquired the developer of GPT-3?" by following edges
from GPT-3 to OpenAI to Microsoft. In contrast, chunk-centric
methods like RAPTOR [ 22] and EraRAG [ 33] organize text chunks
into hierarchical structures based on semantic similarity. They cre-
ate tree-like indexes where leaf nodes contain original text chunks
and parent nodes store increasingly abstract summaries. This de-
sign preserves full contextual information while enabling retrieval
at different levels of granularity.
However, both approaches exhibit fundamental limitations that
restrict their effectiveness. Entity-centric methods suffer from in-
formation loss during the entity extraction process. Named entity
recognition and relation extraction models introduce errors that
compound throughout the system, and the abstraction from text to
entities discards valuable contextual information needed for factual
question answering. Our analysis shows that these methods often
underperform simple dense retrieval on factual QA benchmarks.
Chunk-centric methods face the opposite problem: while they main-
tain high accuracy on factual queries, they cannot capture explicit
relationships between entities scattered across different chunks.
This makes them ineffective for queries requiring logical reason-
ing, such as finding all products affected by a specific supply chain
disruption. The fundamental issue is that existing methods opti-
mize for either contextual completeness or relational structure, but
real-world queries often require both.
Our key insight is that simply combining chunk and entity rep-
resentations in a hybrid graph does not solve the fundamental
problem: both representations remain anchored to their original
textual sources without true knowledge fusion. While knowledge
graphs can connect logically related content that is textually dis-
tant, they cannot integrate the logical information from relations
with the background context from chunks. During retrieval, these
two aspects are still accessed separately through similarity search,
missing the opportunity for synergistic understanding. Consider a
query about "the impact of renewable energy adoption on manufac-
turing industries"—entity-based retrieval might find relationships
between solar panels and factories, while chunk-based retrieval
might return documents about energy costs, but neither captures
the emergent understanding that comes from synthesizing these
perspectives. This motivates our approach: we cluster entities and
chunks together and generate summaries that simultaneously en-
code both relational logic and contextual background, creating new
knowledge representations that transcend the original text.Implementing this knowledge fusion approach presents three
core technical challenges. First, how to construct summaries that
genuinely integrate context and relations rather than simply con-
catenating them. Second, how to leverage these hybrid summaries
during retrieval to access knowledge beyond the original corpus.
The summaries represent emergent understanding that doesn’t ex-
ist in any single source document—the retrieval mechanism must
be able to match queries to these synthesized insights while still
providing access to supporting details. Third, how to efficiently up-
date such complex hierarchical structures when the corpus changes.
Real-world knowledge bases receive continuous updates, but re-
building the entire hierarchy for each new document would be
computationally prohibitive.
To address these challenges, we proposeHyGRAG, a unified
framework with three key innovations. For summary construc-
tion, we design a two-stage approach: first clustering nodes us-
ing graph structure-aware embeddings that capture both textual
similarity and relational connectivity, then prompting LLMs with
structured templates that explicitly request synthesis of contextual
and relational information within each community. This produces
summaries that represent genuine knowledge fusion rather than
simple aggregation. For retrieval, we implement a bi-level mech-
anism that operates across both context and relation dimensions.
Context-aware retrieval searches across all hierarchy levels—from
specific chunks to abstract community summaries—enabling access
to emergent knowledge. Relation-aware retrieval then expands re-
sults by collecting entities from retrieved communities and filtering
their associated triplets, ensuring logical completeness. For dynamic
updates, we develop an attachment-based algorithm where new
content is matched to the most similar community at the appro-
priate abstraction level, with local re-summarization propagating
only along affected paths. This preserves the stability of unrelated
portions while incorporating new knowledge. Experimental results
show thatHyGRAGimproves the average accuracy of multi-hop
reasoning tasks by 9.7%, while maintaining reasonable efficiency.
In summary, we make three contributions:
•We propose hierarchical indexing for Graph RAG that uni-
fies context and relational information through multi-level
abstraction, enabling retrieval across different granularities.
•We develop a retrieval mechanism that combines context-
aware and relation-aware strategies, leveraging community
structures to find information that flat approaches miss.
•We achieve significant performance gains across a range of
tasks, including a 6.2% improvement in Factual Accuracy and
a 9.7% increase in Multi-Hop Reasoning, with improvements
reaching up to 12.2% on the HotpotQA dataset.
2 Related Work
2.1 Graph RAG
RAG [ 5] mitigates hallucinations [ 12] in LLMs by integrating exter-
nal knowledge during generation. RAG involves two coupled steps:
retrieval and generation. Given a query, the system retrieves the
top-k relevant text chunks from a preprocessed external corpus. The
LLM then generates an answer conditioned on both the retrieved
chunks and the query, producing more factual and contextually
appropriate responses.

A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
Vanilla RAG [ 16] improves factual grounding but treats the cor-
pus as unstructured text, limiting inter-document reasoning and
multi-hop inference. To overcome this, recent work incorporates
graph structures into retrieval [ 21], giving rise to graph RAG meth-
ods. These can be categorized as context-aware, focusing on text
chunks, and relation-aware, explicitly leveraging graph structures.
Context-awaremethods enhance generation by retrieving seg-
mented text chunks as context. Vanilla RAG retrieves top-k chunks
based on embedding similarity. Advanced approaches cluster and
summarize chunks to capture high-order information. RAPTOR [ 22]
recursively clusters chunks into hierarchical summaries, while Er-
aRAG [ 33] organizes the corpus into multi-layered summaries using
LSH, handling high-order information in dynamic corpora.
Relation-awaremethods construct textual knowledge graphs
to encode structured relationships. LightRAG [ 7] integrates graph
structures into the text indexing ,and retrieve by high/low-level
keywords. GraphRAG [ 3] cluster communities and summarize them
to provide holistic context. HiRAG [ 11] and ArchRAG [ 29] further
leverage hierarchical graphs to bridge local entity details and global
insights by applying attributed entities. HippoRAG [ 8] and Hip-
poRAG 2 [ 9] enhance multi-hop reasoning by navigating graph
connectivity via Personalized PageRank, improving performance
on complex QA tasks.
2.2 Dynamic Retrieval
Most graph RAG systems assume a static corpus, making updates
costly. LightRAG [ 7] uses a modular retriever to add new documents
dynamically. HippoRAG [ 8] also supports incremental updates,
as it leverages the KG only for retrieval. EraRAG [ 33] introduces
LSH-based clustering, enabling localized updates by re-segmenting
and re-summarizing only affected parts. However, entity-based
community clustering methods still require full reconstruction.
3 Method
Overview.Our proposed framework,HyGRAG, addresses the
fundamental trade-off between context-aware and relation-aware
retrieval through a unified architecture. The system consists of four
key components: (1)Hierarchical Index Structure Construc-
tionconstructs a hierarchical indexing structure with hybrid graph
that captures both chunk-based contextual information and entity-
based relational information; (2)Context and Relation-Aware
Retrievalperforms retrieval across micro and macro granulari-
ties; (3)Retrieval-Augmented Efficient Generationensemble
heirarchical context; (4)Dynamic Knowledge Updatesupports
incremental knowledge integration without full reconstruction.
3.1 Hierarchical Index Structure Construction
This module operates entirely in the offline phase, ensuring no
impact on online retrieval efficiency. We construct a hybrid knowl-
edge graph from raw text that simultaneously captures chunk-based
contextual information and entity-based relational structures.
Text Chunking and Chunk-Level Graph Construction.Given
a corpusD, we first segment it into overlapping text chunks C=
{𝑐1,𝑐2,...,𝑐𝑛}. Each chunk 𝑐𝑖preserves the contextual information
of the original text, enabling inference LLMs to better understandits semantic content. For each chunk, we compute its embedding
representation:
e𝑐
𝑖=LM Embedding(𝑐𝑖),(1)
where we employ BGE-M3 as the embedding model to encode text
chunks into dense vector representations. Based on embedding sim-
ilarity, we construct a chunk-based graph G𝑐=(V𝑐,E𝑐), where
V𝑐=C. We create edges between chunks based on shared enti-
ties rather than shallow embedding similarity. Specifically, if two
chunks𝑐𝑖and𝑐𝑗share more than 𝑙common entities, we create an
edge(𝑐𝑖,𝑐𝑗)∈E𝑐:
(𝑐𝑖,𝑐𝑗)∈E𝑐⇔|E𝑖∩E𝑗|>𝑙,(2)
whereE𝑖andE𝑗are the entity sets extracted from chunks 𝑐𝑖and𝑐𝑗
respectively, and 𝑙is set to 3 in our implementation. This approach
establishes meaningful connections between chunks through shared
entities, providing stronger semantic relationships than surface-
level embedding similarity.
Entity-Level Graph Construction.Then we employ LLMs to
extract knowledge graph triplets from the raw text. Specifically, for
each text chunk 𝑐𝑖, the LLM identifies entities and their relation-
ships, generating a set of triplets:
T𝑖={(ℎ,𝑟,𝑡)|ℎ,𝑡∈E 𝑖,𝑟∈R𝑖},(3)
whereE𝑖andR𝑖are the entity and relation sets extracted from
𝑐𝑖, respectively. By merging triplets from all chunks, we obtain
the global entity set V𝑒=Ð𝑛
𝑖=1E𝑖and relation setE𝑒=Ð𝑛
𝑖=1T𝑖,
forming the entity-based knowledge graphG𝑒=(V𝑒,E𝑒).
Hybrid Graph Construction.To bridge the gap between chunk-
based context and entity-based relations, we connect the two graphs
through membership relationships. For each entity 𝑒∈V𝑒, we
identify the set of text chunks containing that entity:
C(𝑒)={𝑐 𝑖∈C|𝑒is mentioned in𝑐 𝑖}.(4)
We create cross-layer edges E𝑐𝑒connecting entities to their con-
taining chunks:
E𝑐𝑒={(𝑒,𝑐)|𝑒∈V𝑒,𝑐∈C(𝑒)}.(5)
The final hybrid knowledge graph is defined as:
Ghybrid=(V𝑐∪V𝑒,E𝑐∪E𝑒∪E𝑐𝑒).(6)
This graph contains two types of nodes (chunk nodes and entity
nodes) and three types of edges (chunk-to-chunk edges, entity-to-
entity edges, and cross-layer edges).
Hierarchical Indexing.While the hybrid graph enriches infor-
mation representation, it also increases the complexity of online
retrieval. To efficiently retrieve both relevant contextual and re-
lational information simultaneously, we propose to construct a
hierarchical tree-structured index for the hybrid graph.
First, we use Cleora to generate structure-aware embeddings for
all nodes𝑣∈V𝑐∪V𝑒in the hybrid graph:
z𝑣=Cleora(G hybrid,𝑣).(7)
Then, we employ hyperplane-based Locality-Sensitive Hashing
(LSH) for efficient clustering. For each node embeddingz 𝑣∈R𝑑,
we randomly sample 𝑘hyperplanes{h1,..., h𝑘}and compute its
hash code:
hash(z𝑣)=[sign(z 𝑣·h1),...,sign(z 𝑣·h𝑘)](8)

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Haoyang Zhong et al.
KGrelation
Belong to
Overlaping
Chunk NodeEntity Node
Documents
Chunks
Hybrid GraphNER
OpenieOffline Indexing
Online Retrieval 
& GenerationCleora
LSH
Community 
Node
Query
Retrieved chunksRetrieved enititiesRetrieved communities
Filtering
LLMHeirarchical Indexing Structure
Relation
ExtractionRelation
IntegrationQuery
Context -aware RetrievelRelation -aware Retrievel
Node
CollectionStructured
Prompt
TemplateRelation
Text
AnswerRetrieval
from
Indexing 
StructureContaining Hybrid EdgesMulti -layer
Community
Figure 2: Overall architecture ofHyGRAG.
Nodes with similar hash codes (small Hamming distance) are as-
signed to the same bucket. For each bucket 𝐵, if its size satisfies
𝑆min≤|𝐵|≤𝑆 max, it is retained; otherwise, split or merge opera-
tions are performed:
𝐵adjusted = 
Split(𝐵)if|𝐵|>𝑆 max
Merge(𝐵,𝐵 neighbor)if|𝐵|<𝑆 min
𝐵otherwise(9)
For each adjusted bucket (forming a community) 𝐶, we generate
summary representations through a two-step process:
𝑡𝐶=LLM summarize({𝑣|𝑣∈𝐶}),(10)
𝑠𝐶=LM Embedding(𝑡𝐶),(11)
where we utilize Llama3.1-8B-Instruct to generate concise tex-
tual summaries 𝑡𝐶that capture the semantic essence of nodes within
each community, and then employ BGE-M3 to encode these sum-
maries into dense vector representations𝑠 𝐶.
We treat summary nodes as nodes in the next layer and re-
cursively apply the above hashing, bucketing, and summarization
process up to a predefined number of layers. Let Lℓdenote the node
set at layerℓ, whereL0=V𝑐∪V𝑒represents the leaf nodes. The
hierarchical construction continues until the layer index ℓreaches
the preset maximum depth 𝐿, resulting in an 𝐿-layer hierarchical
index tree:
Tindex=L 0,L1,...,L𝐿.(12)
This hierarchical tree serves as an efficient retrieval database dur-
ing the dual-stage retrieval phase, where leaf nodes correspond
to original nodes from the hybrid graph and upper-layer nodes
contain semantic summaries at coarser granularities, supporting
multi-scale context and relation-aware retrieval.3.2 Context and Relation-Aware Retrieval
During the online retrieval phase, we employ a bi-level approach
that captures information at both context and relation levels to
provide comprehensive and contextually rich results.
Query Encoding.Given a user query 𝑞, we first encode it using
the same embedding model employed during indexing:
q=LM Embedding(𝑞),(13)
where we utilize BGE-M3 to generate the query embeddingq ∈R𝑑
that serves as the basis for similarity computation across all retrieval
levels.
Context-Aware Retrieval.From the hierarchical indexing struc-
tureTindex containing community, chunk, and entity nodes, we first
perform context-aware retrieval by computing similarity scores
across all node types. We retrieve the top- 𝑘most similar nodes for
each type:
Pretrieved =TopK({𝐶∈𝐿Ø
ℓ=1Lℓ|sim(q,𝑠𝐶)},𝑘),(14)
Cretrieved =TopK({𝑐∈V𝑐|sim(q,e𝑐
𝑐)},𝑘),(15)
Econtext
retrieved=TopK({𝑒∈V𝑒|sim(q,z𝑒)},𝑘).(16)
These retrievals capture content most relevant to the query from
a contextual perspective across different granularities.
Relation-Aware Retrieval.To capture logical relationships, we
construct a comprehensive entity set by combining entities from
retrieved communities and context-based entity retrieval:
Eall=Econtext
retrieved∪Ø
𝐶∈P retrieved{𝑒|𝑒∈V𝑒,(𝑒,𝐶)∈E𝑐𝑒}.(17)

A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
We then extract all relations connected to these entities from
leaf nodes:
Rall={(ℎ,𝑟,𝑡)∈E𝑒|ℎ∈E all∨𝑡∈E all}.(18)
Since|Rall|can be large, potentially introducing noise and LLM
overhead, we filter by computing triplet embeddings and selecting
the top-𝑘most relevant:
Rretrieved =TopK({(ℎ,𝑟,𝑡)∈R all|z(ℎ,𝑟,𝑡)=LM Embedding(ℎ⊕𝑟⊕𝑡)},𝑘),
(19)
where⊕denotes concatenation.
Integrated Retrieval Strategy.The final retrieval combines context-
aware and relation-aware perspectives, yielding four sets of 𝑘ele-
ments each:
Rfinal={P retrieved,Cretrieved,Econtext
retrieved,Rretrieved},(20)
providing community summaries for high-level understanding,
chunks for detailed context, entities for key concepts, and rela-
tions for logical connections.
Computational Complexity Analysis.Similar to other retrieval-
augmented approaches,HyGRAGemploys a FAISS vector store
with HNSW indexing to efficiently retrieve the top- 𝑘most similar
items. Let𝑁denote the total number of indexed nodes and 𝑑the
embedding dimension. The HNSW-based search achieves logarith-
mic complexity for each index: O(log𝑁𝑖·𝑑),and retrieving from
entity-, chunk-, and community-level indexes results in:
O((log𝑁𝑒+log𝑁𝑐+log𝑁𝑝)·𝑑)=O(log𝑁·𝑑).(21)
For each retrieved entity, relation extraction and relevance scoring
introduce an additional cost of O(𝑘𝑒·¯𝑑·𝑑) , where ¯𝑑is the average
node degree. The overall retrieval complexity is thus:
O(log𝑁·𝑑+𝑘 𝑒·¯𝑑·𝑑),(22)
which maintains sub-linear scaling with respect to graph size while
enabling multi-granularity and relation-aware retrieval.
Compared with traditional chunk-aware RAG systems that only
perform a single chunk-level retrieval O(log𝑁𝑐·𝑑), our method
introduces additional but lightweight costs from entity and com-
munity retrievalsO((log𝑁 𝑒+log𝑁𝑝)·𝑑+𝑘𝑒·¯𝑑·𝑑).
3.3 Retrieval-Augmented Efficient Generation
Given the retrieved nodes from bi-level retrieval, we extract and
organize information for LLM generation. For community nodes
𝐶∈ P retrieved , we obtain their summaries 𝑡𝐶. For entity nodes
𝑒∈Econtext
retrieved, we extract their textual representations. For relation
triplets(ℎ,𝑟,𝑡)∈R retrieved , we preserve their structured format. For
chunk nodes𝑐∈C retrieved , we directly use their textual content.
We combine these four information types using a structured
prompt template:
Context=P({𝑡 𝐶}𝐶∈P retrieved,Econtext
retrieved,Rretrieved,Cretrieved),(23)
wherePorganizes community summaries, entities, relation triplets,
and chunk contexts hierarchically.
The final response is generated by:
𝑦=LLM generate(𝑞,Context),(24)leveraging community summaries for high-level understanding,
entities for key concepts, relation triplets for logical reasoning, and
chunk contexts for detailed background information.
3.4 Dynamic Knowledge Update
In real-world scenarios, knowledge corpora evolve continuously,
requiring efficient update mechanisms. Our clustering-based hier-
archical structure enables fast integration of new content without
full graph reconstruction.
Update Processing.Given new text content 𝑑new, we first segment
it into chunksCnew={𝑐1
new,...,𝑐𝑚
new}and extract knowledge graph
triplets:
Tnew=𝑚Ø
𝑖=1{(ℎ,𝑟,𝑡)|ℎ,𝑡∈Enew
𝑖,𝑟∈Rnew
𝑖}.(25)
We generate a summary representation for the new content:
𝑡new=LLM summarize(Cnew∪Tnew),(26)
snew=LM Embedding(𝑡new).(27)
Hierarchical Attachment.We traverse the hierarchical index
from bottom to top. Starting at layer L1, we find the most similar
community:
𝐶∗=arg max
𝐶∈L 1sim(s new,𝑠𝐶).(28)
Ifsim( snew,𝑠𝐶∗)>𝜏 attach , we attach the new content to 𝐶∗. Oth-
erwise, we proceed to the next layer:
ℓ∗=min{ℓ|∃𝐶∈L ℓ:sim(s new,𝑠𝐶)>𝜏 attach}.(29)
Upon attachment at layer ℓ∗, we update all ancestor community
summaries along the path to the root:
∀𝑗∈{ℓ∗,...,𝐿}, 𝑡 𝐶𝑗←LLM summarize(Children(𝐶 𝑗)),(30)
𝑠𝐶𝑗←LM Embedding(𝑡𝐶𝑗).(31)
Additionally, we establish connections between new nodes and
existing leaf nodes following the original hybrid graph construction
rules, creating edges when entities are shared or semantic similarity
exceeds thresholds. Since all updates occur offline, online retrieval
efficiency remains unaffected.
4 Experiments
In this section, we answer the following questions to validate the
effectiveness of our method:
•RQ1.How doesHyGRAGperform compared with existing
baselines on different type of static QA tasks?
•RQ2.How efficient is theHyGRAGapproach?
•RQ3.How robust isHyGRAGin corpus expansion?
•RQ4.What is the quality of our communities and relations,
and how does it impact the performance of RAG?
4.1 Experimental Setup
4.1.1 Datasets and Metrics.For static QA, we adopt five datasets:
PopQA[ 19] (factual accuracy),MuSiQue[ 27], andHotpotQA
[32] (multi-hop reasoning), andMultiHop-RAG[ 26],QuALITY
[20] (reading comprehension). For evaluation, static datasets report
AccuracyandRecall(QuALITY uses Accuracy only).

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Haoyang Zhong et al.
Table 1: Overall QA results(Accuracy and Recall) on static RAG query datasets using Llama-3.1-8B-Instruct.
Method Factual Accuracy Reading Comprehension Multi-Hop Reasoning
PopQA QuALITY MultiHop-RAG MuSiQue HotpotQA
Accuracy Recall Accuracy Accuracy Recall Accuracy Recall Accuracy Recall
Zero-shot 15.37 5.62 33.54 27.93 20.84 8.76 19.19 27.36 32.97
CoT 16.03 5.58 33.83 28.13 21.01 8.53 18.95 27.77 38.88
VanillaRAG 66.98 31.64 56.38 58.49 37.52 32.3 43.92 60.97 64.24
RAPTOR 66.33 30.98 58.08 58.72 37.88 31.17 43.47 OOT OOT
RAPTOR-K 66.26 30.86 58.34 57.82 37.45 31.67 43.91 61.24 65.19
EraRAG 61.40 29.9460.7958.67 37.89 26.70 38.93 61.03 65.22
L-LightRAG 67.05 39.97 48.05 53.05 37.38 30.63 43.66 OOT OOT
G-LightRAG 23.52 11.26 30.53 52.15 35.42 16.01 27.73 OOT OOT
H-LightRAG 47.46 25.91 38.60 52.86 36.84 18.73 29.41 OOT OOT
HippoRAG 55.18 23.35 46.10 58.88 31.36 20.00 32.54 51.89 56.31
HippoRAG2 68.12 38.77 45.98 61.66 33.72 30.23 39.23 OOT OOT
L-GraphRAG 54.82 30.89 50.01 53.01 36.78 14.57 28.72 OOT OOT
HiRAG 53.18 30.47 55.83 54.00 38.09 26.20 40.70 OOT OOT
ArchRAG 30.81 14.04 37.77 60.64 33.44 14.27 29.22 OOT OOT
HyGRAG72.34 43.5159.49 73.79 (65.41) 44.05 (39.44) 35.87 48.06 68.72 70.79
+Inference 72.38 43.62 59.79 74.33 44.09 36.6 48.36 69.10 71.08
4.1.2 Baseline.We compare our approach with three categories
of RAG baselines: (1) LLM-only:ZeroShot,Chain-of-Thought
(CoT)[ 14]; (2) Context-aware:VanillaRAG[ 16],RAPTOR[ 22]
(GMM/K-means clustering algorithm), andEraRAG[ 33]; (3) Relation-
aware:L-GraphRAG[ 3](local mode of Microsoft GraphRAG),Hip-
poRAG[ 8],HippoRAG2[ 9],HiRAG[ 11],ArchRAG[ 29] andLigh-
tRAG[ 7]. Variants of LightRAG, namely Local, Global, and Hybrid,
are denoted asL/G/H-LightRAGfor brevity.
4.1.3 Implementation Details.All static-query experiments use
Llama-3.1-8B-Instruct[6] as backbone under the vLLM[15] engine
with greedy decoding and top-k=5 retrieval. For a fair comparison,
we maintain retrieval sizes across all node types, i.e., entity nodes=
community + chunk nodes=relations=top-𝑘 . Embeddings are
generated by BGE-M3[ 2], a SOTA embedding model that supports
both multilingual and multi-granularity retrieval, splitting text into
1,200-token chunks with 100-token overlap. Baselines follow the
framework[ 35] or official code with default hyperparameters. Runs
exceeding two days are marked asOOT.
4.2 Static QA Performace
To answer the question RQ1, we present the experimental results
of question answering (QA) on various types of static queries, in-
cluding factual accuracy, multi-hop reasoning, and reading com-
prehension tasks. Our method is systematically compared with a
range of baseline models to assess its effectiveness across different
reasoning scenarios. Furthermore, the statistics of different datasets
are summarized in the Appendix C.
4.2.1 QA results.As shown in Table 1, different categories of base-
line methods exhibit substantial variations in static QA perfor-
mance. LLM inference-only approaches (Zero-shot and CoT) per-
form poorly across all datasets, with accuracies significantly lower
than retrieval-based methods. This indicates that relying solely onthe parametric knowledge of LLMs is insufficient for factual and
reasoning-intensive tasks.
Among context-level methods, VanillaRAG, RAPTOR, and Er-
aRAG achieve comparable performance on factual accuracy and
multi-hop reasoning datasets. This suggests that the advantage of
high-order community structures is not always benificial, and the
effectiveness largely depends on whether multi-hop content is ag-
gregated into the same community during the offline indexing stage.
On the QuALITY dataset, however, RAPTOR and EraRAG exhibit
clear advantages, with EraRAG achieving the best overall perfor-
mance. We attribute this to EraRAG’s retrieval strategy, which
enforces the selection of one non-leaf node and four leaf nodes,
thereby reducing irrelevant information and facilitating semantic
understanding by the LLM. Overall, RAPTOR slightly outperforms
RAPTOR-K and EraRAG, mainly due to its use of a superior but ex-
pensive clustering method, though the improvements are marginal.
For relation-level methods, the LightRAG variants demonstrate
divergent strengths. L-LightRAG achieves the highest recall (39.97%)
on PopQA, showing that relation-aware retrieval can effectively
support LLMs in factual accuracy tasks. In contrast, G-LightRAG
and H-LightRAG perform relatively poorly, likely because global
higher-level concepts are less suited for highly specific queries.
ArchRAG and HiRAG, as improved versions of L-GraphRAG, re-
fines reasoning path generation by leveraging semantically sim-
ilar summary entities. HiRAG achieves better performance than
L-GraphRAG in multi-hop reasoning, though still leaving room for
improvement. HippoRAG and its improved version, HippoRAG2,
improving retrieval performance through relations, exhibit com-
petitive results on tasks such as MultiHop-RAG and PopQA but
consistently fall short of our method in terms of overall accuracy.
Overall,HyGRAGachieves the best or near-best performance
across all tasks. On PopQA, it reaches 72.34% accuracy and 43.51%
recall, outperforming other methods by 6.2% and 8.9%, respectively.
On the QuALITY dataset, it achieves the second-best result. In
multi-hop reasoning tasks,HyGRAGattains 65.41% accuracy on

A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
Figure 3: Comparison of query efficiency.
MultiHop-RAG, a 6.0% improvement over the strongest baseline(the
metric in parentheses denotes results without explicitly prompt-
ing the model to state “Insufficient Information” when possible),
and shows stable advantages on MuSiQue and HotpotQA, with
an average gain of approximately 11.1%. With the addition of in-
ference augmentation (+Inference), which add a rule in prompt to
encourage the model to leverage our domain knowledge for rea-
soning, MuSiQue and HotpotQA further improve by 2.0% and 0.4%,
respectively. These results demonstrate thatHyGRAGconsistently
delivers robustness and superior effectiveness across factual QA,
reading comprehension, and multi-hop reasoning tasks through
a synergistic integration of relation and context. In addition, we
record in the Appendix D the results of further analyses using
different dense retrievers and replacing the base LLM with Qwen3-
8B[31]. The detailed results are shown in Table 10 and Table 9. We
observe that across various embedding models,HyGRAGconsis-
tently maintains stable and competitive performance, indicating
that its advantages are not limited to a specific dense retriever.
When replacing the base LLM with the newer model Qwen3-
8B[31] (using a consistentno-thinkingmode), the overall trend
remains consistent with previous analyses. Notably, we observe an
unexpected performance drop of context-aware methods on the
MuSiQue and HotpotQA datasets after replacing the base LLM. In
contrast,HyGRAGmaintains competitive results, achieving up to
10% higher accuracy than other relation-aware approaches. This
demonstrates its robustness and adaptability across different model
architectures and reasoning paradigms.
4.2.2 Efficiency ofHyGRAG.To answer the question RQ2, we
compare the time cost and token usage ofHyGRAGwith those of
other baseline methods. As shown in Figure 3,HyGRAGdemon-
strates significant time and cost efficiency especially among relation-
aware methods for online queries. However, compared with context-
aware methods, the cost ofHyGRAGis higher.
Figure 4: Corpus expansion performance.
Figure 5: Corpus expansion indexing cost.
4.2.3 Robustness to Corpus Expansion.As RAG systems are in-
creasingly deployed in real-world applications, they must become
more adaptable to scenarios of continuous learning where the re-
trieval corpus keeps expanding. To answer the question RQ3, in
order to evaluate the ability ofHyGRAGto handle incremental
corpus insertions, we designed an experiment that divides the initial
corpus into different proportions and then incrementally inserts the
remaining corpus as updates. We then tested the QA performance
under varying proportions of incremental corpus, and the results
are shown in Figure 4.
From the results, we observe that the larger the initial corpus
proportion, the better the final QA performance. This is mainly
because, compared to static construction, incremental corpus in-
sertion leads to a slight decrease in community quality. However,
this decrease is minor, around 1–2%, which shows the capability
ofHyGRAGto handle continual learning scenarios effectively.
In addition, we measured the reconstruction time and token con-
sumption required by various baseline methods when 20% of the

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Haoyang Zhong et al.
Table 2: Results of ablation study.
MultiHop-RAG QuALITY PopQA
Acc Rec Acc Acc Rec
HyGRAG(Full)65.41 39.4459.49 72.34 43.51
w/o entity&relation 64.35 39.08 60.2070.80 41.30
w/o community 65.02 39.01 58.75 71.92 43.18
w/o chunk 52.19 35.89 37.82 51.25 26.81
vanillaRAG 58.49 37.52 56.38 66.98 31.64
Table 3: Comparative performance and token cost against
combining HiRAG and RAPTOR.
Method MultiHop-RAG MuSiQue
Acc Rec Tokens Acc Rec Tokens
HiRAG + RAPTOR 62.36 37.39 7170.7 31.30 43.80 3277.2
+ prompt 70.27 41.37 7280.2 32.47 44.58 3768.5
HyGRAG73.79 44.05 5030.3 35.87 48.06 1720.0
corpus is inserted, as shown in Figure 5. The results show that our
proposedHyGRAGachieves favorable efficiency and competitive
performance, indicating its strong practicality in dynamic corpus.
4.3 Detailed Analysis
To answerRQ4, we conduct detailed analyses to investigate how
each component ofHyGRAGcooperatively affects the overall
performance of the system.
4.3.1 Ablation Study.Table 2 presents the results of our ablation
studies conducted on multiple datasets. To assess the individual con-
tributions of each retrieval strategy, we progressively removed spe-
cific components and examined the resulting performance changes.
As shown in Table 2, removing the chunk-level structure leads to
the most pronounced degradation across all datasets, highlighting
its crucial role in providing fundamental contextual grounding and
factual support. Similarly, eliminating entity and relation informa-
tion causes a decline in accuracy, confirming the importance of
relational reasoning for both factual and multi-hop question an-
swering tasks. Interestingly, on the quality dataset, this removal
yields a slight performance improvement. We attribute this to the
fact that, in reading comprehension tasks—particularly for smaller
LLMs—explicit relational cues may sometimes mislead the model,
whereas context-aware semantic signals are more essential when
the answer requires inference rather than direct textual matching.
In contrast, removing the community-level representation results
in a relatively mild yet stable decrease, indicating that commu-
nity structures primarily contribute to aggregating query-relevant
higher-order knowledge, providing semantic summarization, and
optimizing relation retrieval. Overall, these findings confirm that
HyGRAGbenefits substantially from the joint design of context-
aware and relation-aware mechanisms.
4.3.2 Comparative Analysis.To further examine the complemen-
tary effects of thecontext-awareandrelation-awarecomponents,
we conducted comparative experiments. Specifically, we combinedthe retrieved chunks and communities from RAPTOR (the best-
performing context-aware method) with the entities and relations
from HiRAG (the modified method of MS GraphRAG in multi-hop
reasoning path generation). We then compared the results with and
without applying the same system prompt used inHyGRAG.
As shown in Table 3, combining HiRAG and RAPTOR yields mod-
erate improvements after applying theHyGRAG-style prompt but
still underperforms our model in both accuracy and recall. Notably,
HyGRAGachieves superior or comparable performance with sub-
stantially fewer tokens, demonstrating its efficiency in balancing
reasoning depth and retrieval precision. These results demonstrate
that the integrated design ofHyGRAGis not a mere concatenation
of relation and context representations, but rather an organic and
unified framework that jointly models both aspects, enabling LLMs
to more effectively leverage contextual and relational knowledge
during reasoning.
4.3.3 Case Study.To further illustrate the qualitative advantages
of our framework, we present representative examples from the
PopQA and MuSiQue datasets. For each dataset, we compareHy-
GRAGwith the best-performing baseline method. Detailed case
studies of model responses are provided in the Appendix A. The
results demonstrate thatHyGRAGeffectively integrates relevant
entities, relations, contextual information, and community-level
knowledge to assist the model in constructing coherent reasoning
chains and producing factually consistent answers. For instance,
in the MuSiQue multi-hop reasoning task, baseline models such
as LLightRAG and VanillaRAG fail to retrieve the correct passages
through semantic search or to establish the connection between
Jan Klapáč’s birthplace and the target entity Prague Castle. In con-
trast,HyGRAGsuccessfully leverages relational and contextual
information from both the chunk and community levels to capture
the latent associations and infer the correct answer. Similarly, in a
factual query from PopQA, under a setting involving multiple per-
son entities, the model accurately identifies the target entity Paul
Walker and retrieves the supporting evidence from key relation
that he was born in Kilwinning. These cases collectively illustrate
thatHyGRAGnot only achieves unified modeling of relational
and contextual knowledge but also enhances entity disambiguation
and relational consistency during reasoning, thereby improving
both factual accuracy and interpretability.
5 Conclusion
This work introduces a unified framework with context and relation-
aware retrieval.HyGRAGaddresses the problem that entity-centric
and chunk-centric methods operate on representations anchored
to original text. By clustering hybrid nodes and generating commu-
nities integrating contextual and relational information, we create
summary representations beyond source documents. Our bi-level
retrieval enables access to insights neither approach could achieve
independently. Furthermore,HyGRAG’s attachment-based update
capability ensures efficient incorporation of new information. Over-
all,HyGRAGshows 9.7% average improvements in multi-hop
reasoning while maintaining efficiency for dynamic corpora.
Acknowledgments
This work is supported by NSFC (No. 62322606, No. 62441605).

A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
References
[1]@articlesun2025mlnc, author = Yifei Sun and Zemin Liu and Bryan Hooi and
Yang Yang and Rizal Fathony and Jia Chen and Bingsheng He, ... . 2025. Multi-
Label Node Classification with Label Influence Propagation. InInternational
Conference on Learning Representations.
[2]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.
2024. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity
Text Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216 [cs.CL]
https://arxiv.org/abs/2402.03216
[3]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2025. From Local to Global: A Graph RAG Approach to Query-Focused
Summarization. arXiv:2404.16130 [cs.CL] https://arxiv.org/abs/2404.16130
[4]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A Survey on RAG Meeting LLMs: Towards
Retrieval-Augmented Large Language Models. arXiv:2405.06211 [cs.CL] https:
//arxiv.org/abs/2405.06211
[5]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[6]Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek
Kadian, et al .2024. The Llama 3 Herd of Models. arXiv:2407.21783 [cs.AI]
https://arxiv.org/abs/2407.21783
[7]Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. arXiv:2410.05779 [cs.IR] https:
//arxiv.org/abs/2410.05779
[8]Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2025. HippoRAG: neurobiologically inspired long-term memory for large lan-
guage models. InProceedings of the 38th International Conference on Neural Infor-
mation Processing Systems(Vancouver, BC, Canada)(NIPS ’24). Curran Associates
Inc., Red Hook, NY, USA, Article 1902, 38 pages.
[9]Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From RAG to Memory: Non-Parametric Continual Learning for Large Language
Models. arXiv:2502.14802 [cs.CL] https://arxiv.org/abs/2502.14802
[10] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao.
2025. GRAG: Graph Retrieval-Augmented Generation. arXiv:2405.16506 [cs.LG]
https://arxiv.org/abs/2405.16506
[11] Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu Pan, Yongqiang Chen, Kaili
Ma, Hongzhi Chen, and James Cheng. 2025. Retrieval-Augmented Generation
with Hierarchical Knowledge. arXiv:2503.10150 [cs.CL] https://arxiv.org/abs/
2503.10150
[12] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A Survey on Hallucination in Large Language Models: Principles,
Taxonomy, Challenges, and Open Questions.ACM Transactions on Information
Systems43, 2 (Jan. 2025), 1–55. doi:10.1145/3703155
[13] Thomas N. Kipf and Max Welling. 2016. Semi-Supervised Classification with
Graph Convolutional Networks.CoRRabs/1609.02907 (2016). arXiv:1609.02907
http://arxiv.org/abs/1609.02907
[14] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke
Iwasawa. 2022. Large language models are zero-shot reasoners. InProceedings of
the 36th International Conference on Neural Information Processing Systems(New
Orleans, LA, USA)(NIPS ’22). Curran Associates Inc., Red Hook, NY, USA, Article
1613, 15 pages.
[15] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Mem-
ory Management for Large Language Model Serving with PagedAttention. In
Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.
[16] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InProceedings of the 34th International Conference
on Neural Information Processing Systems(Vancouver, BC, Canada)(NIPS ’20).
Curran Associates Inc., Red Hook, NY, USA, Article 793, 16 pages.
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. arXiv:2005.11401 [cs.CL] https://arxiv.org/abs/
2005.11401
[18] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin
Mao, and Jian Guo. 2025. Think-on-Graph 2.0: Deep and Faithful Large Language
Model Reasoning with Knowledge-guided Retrieval Augmented Generation.
arXiv:2407.10805 [cs.CL] https://arxiv.org/abs/2407.10805
[19] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Han-
naneh Hajishirzi. 2023. When Not to Trust Language Models: Investigating Effec-
tiveness of Parametric and Non-Parametric Memories. arXiv:2212.10511 [cs.CL]https://arxiv.org/abs/2212.10511
[20] Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang,
Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, and
Samuel R. Bowman. 2022. QuALITY: Question Answering with Long Input Texts,
Yes! arXiv:2112.08608 [cs.CL] https://arxiv.org/abs/2112.08608
[21] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong,
Yan Zhang, and Siliang Tang. 2024. Graph Retrieval-Augmented Generation: A
Survey. arXiv:2408.08921 [cs.AI] https://arxiv.org/abs/2408.08921
[22] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher D. Manning. 2024. RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval. InInternational Conference on Learning Representations
(ICLR).
[23] Yifei Sun, Haoran Deng, Yang Yang, Chunping Wang, Jiarong Xu, Renhong Huang,
Linfeng Cao, Yang Wang, and Lei Chen. 2022. Beyond Homophily: Structure-
aware Path Aggregation Graph Neural Network. InProceedings of the Thirty-First
International Joint Conference on Artificial Intelligence, IJCAI-22, Lud De Raedt
(Ed.). International Joint Conferences on Artificial Intelligence Organization,
2233–2240. doi:10.24963/ijcai.2022/310 Main Track.
[24] Yifei Sun, Yang Yang, Xiaojuan Feng, Zijun Wang, Haoyang Zhong, Chunping
Wang, and Lei Chen. 2025. Handling Feature Heterogeneity with Learnable
Graph Patches. InKnowledge Discovery and Data Mining. https://dl.acm.org/doi/
10.1145/3690624.3709242
[25] Yifei Sun, Qi Zhu, Yang Yang, Chunping Wang, Tianyu Fan, Jiajun Zhu, and Lei
Chen. 2024. Fine-Tuning Graph Neural Networks by Preserving Graph Generative
Patterns. InProceedings of the AAAI Conference on Artificial Intelligence, Vol. 38.
9053–9061.
[26] Yixuan Tang and Yi Yang. 2024. MultiHop-RAG: Benchmarking Retrieval-
Augmented Generation for Multi-Hop Queries. arXiv:2401.15391 [cs.CL] https:
//arxiv.org/abs/2401.15391
[27] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
arXiv:2108.00573 [cs.CL] https://arxiv.org/abs/2108.00573
[28] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder,
and Furu Wei. 2024. Multilingual E5 Text Embeddings: A Technical Report.
arXiv:2402.05672 [cs.CL] https://arxiv.org/abs/2402.05672
[29] Shu Wang, Yixiang Fang, Yingli Zhou, Xilin Liu, and Yuchi Ma. 2025. ArchRAG:
Attributed Community-based Hierarchical Retrieval-Augmented Generation.
arXiv:2502.09891 [cs.IR] https://arxiv.org/abs/2502.09891
[30] Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong,
Xiao Huang, and Jinsong Su. 2025. When to use Graphs in RAG: A Comprehensive
Analysis for Graph Retrieval-Augmented Generation. arXiv:2506.05690 [cs.CL]
https://arxiv.org/abs/2506.05690
[31] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, et al .2025.
Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https://arxiv.org/abs/2505.
09388
[32] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. arXiv:1809.09600 [cs.CL]
https://arxiv.org/abs/1809.09600
[33] Fangyuan Zhang, Zhengjun Huang, Yingli Zhou, Qintian Guo, Zhixun Li, Wen-
sheng Luo, Di Jiang, Yixiang Fang, and Xiaofang Zhou. 2025. EraRAG: Effi-
cient and Incremental Retrieval Augmented Generation for Growing Corpora.
arXiv:2506.20963 [cs.IR] https://arxiv.org/abs/2506.20963
[34] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang,
Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou.
2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through
Foundation Models. arXiv:2506.05176 [cs.CL] https://arxiv.org/abs/2506.05176
[35] Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, Runyuan
He, Yongwei Zhang, Sicong Liang, Xilin Liu, Yuchi Ma, and Yixiang Fang.
2025. In-depth Analysis of Graph-based RAG in a Unified Framework.
arXiv:2503.04338 [cs.IR] https://arxiv.org/abs/2503.04338
A Case Study
As shown in Table 4, the baselines fail to establish the correct
reasoning chain required to answer the question. LLightRAG re-
trieves partial contextual evidence (i.e., Jan Klapáč’s birthplace)
but lacks the relational linkage to the target entity, while Vanil-
laRAG produces an entirely irrelevant response due to the absence
of structured retrieval. In contrast,HyGRAGsuccessfully inte-
grates entity-level, relational, and community-level information.
It first identifies the entity Jan Klapáč and his birthplace (Prague),
then connects this with the relation located in and the community
summary mentioning Prague Castle. This multi-level reasoning

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Haoyang Zhong et al.
Table 4: Case Study on MuSiQue.
Category Multi-hop reasoning
Query What is the name of the castle in the city Jan
Klapáč was born in? Answer: Prague Castle
LLightRAGJan Klapáč’s Birthplace:
Jan Klapáč was born in Prague , Protectorate of Bohemia
and Moravia.
Castle in Prague:
However, the data does not provide information about
a specific castle in Prague where Jan Klapáč was born.
VanillaRAG Unfortunately, the given context does not mention Jan
Klapáč. However, based on the information provided,
we can see that Casimir Pulaski, a Polish general and
hero of the American Revolutionary War, was born in
Warsaw.
HyGRAGAnalysis:
To answer the user’s question, we need to find the con-
nection between Jan Klapáč’s birthplace and the castle
mentioned in the context.
From the **Most Relevant Entities**, we see that Prague
Castle is the most relevant entity, but it’s not directly
related to Jan Klapáč’s birthplace.
However, from the **Key Relationships**, we find that
Jan Klapáč was born in Prague (relationship 1: jan klap
–[born in]–> prague).
From the **Most Relevant Documents**, we see that
Document 1 mentions Jan Klapáč’s birthplace as Prague,
Protectorate of Bohemia and Moravia.
Now, let’s look for a connection between Prague and
a castle. From the **Key Relationships**, we find that
Prague Castle is located in the Czech Republic (rela-
tionship 3: prague castle –[located in]–> czech repub-
lic). ... From the **Community Summary**, we see that
Prague Castle is a central entity associated with the ar-
chitectural styles of Baroque and Mannerism...
So, the name of the castle in Prague is Prague Castle .
enables the model to recover the implicit connection between the
birthplace and the target entity, ultimately leading to the correct
answer (Prague Castle). This case illustrates thatHyGRAGeffec-
tively captures and organizes cross-entity relations and contextual
dependencies, allowing the LLM to perform more accurate and
interpretable multi-hop reasoning.
Table 5 presents a case from the PopQA dataset illustratingHy-
GRAG’s superiority in handling entity ambiguity. The query “In
what city was Paul Walker born?” involves multiple homonymous
entities. Baseline models like LLightRAG and RAPTOR confuse
different individuals, producing inconsistent or irrelevant answers
(e.g., Balwyn, Colac, or Glendale). In contrast,HyGRAGcom-
bines entity–relation reasoning with chunk-level to filter out un-
related entities and consolidate consistent evidence. It correctly
identifies Kilwinning and North Ayrshire—matching the gold an-
swer—demonstrating that its relational and hierarchical design
substantially improves factual accuracy and interpretability.Table 5: Case Study on PopQA.
Category Factual accuracy
Query In what city was Paul Walker born? Answer:
Kilwinning | The Crossroads of Ayrshire
LLightRAG Paul Walker was born in Balwyn or Colac, Victoria, Aus-
tralia. According to the data provided, both Balwyn and
Colac are listed as places where Walker was born, with
Balwyn being noted as a suburb in Victoria, Australia,
and Colac as a town in the same region. The informa-
tion suggests that Walker’s birthplace is associated with
these two locations, though it does not definitively state
which one is the exact place of birth. This ambiguity is
reflected in the sources, which mention that Walker was
born in Balwyn or Colac in 1927.
RAPTOR Based on the provided information, there are multiple
individuals named Paul Walker, and each has a different
birthplace.
However, the most well-known Paul Walker, the actor,
was born in Glendale, California, on September 12, 1973.
If you are referring to Paul Baxendale-Walker, the Arctic
explorer, he was born in Shrewsbury, England, on March
9, 1966.
If you are referring to the other Paul Walker, the one
born in Altoona, Pennsylvania, or Memphis, the infor-
mation provided is not as detailed as the other two, but
it can be concluded that he was born in either Altoona,
Pennsylvania, or Memphis, on March 15, 1968, or in Al-
toona, Pennsylvania, on an unspecified date.
HyGRAGAnalysis of Evidence and Reasoning: ...
Key Relationships: 1. **Paul Walker –[was born in]–>
England** 2. **Paul Walker –[born in]–> Scotland** 3.
**Paul Walker –[was born in]–> Shrewsbury** 4. **Paul
Walker –[born in]–> Kilwinning ** 5. **Paul Walker –
[born in]–> North Ayrshire**
These relationships suggest multiple possible birthplaces
for Paul Walker, including ...
Documents: - **Document 1** states: "He was born in
Shrewsbury, England on 09 March 1966. " ... - **Document
5** states: "Walker was born in Franklin, Georgia, on
October 22, 1935."
These documents provide conflicting birthplaces for
**Paul Walker**, with different cities and even different
years of birth. ...
Conclusion: Based on the **Key Relationships**, the
most reliable answers are: - **Shrewsbury, England**
- **Kilwinning ** - **North Ayrshire**
These are all in **Scotland** or **England**, and they are
explicitly stated in the **Key Relationships**.
However, the **Documents** provide conflicting infor-
mation: - **Document 1** says **Shrewsbury, England**
... This suggests that there may be multiple individuals
named **Paul Walker**, or that the documents refer to
different people with the same name....
B Algorithm Details
The index construction process is illustrated in Algorithm 1. Subse-
quently, the logic of the retrieval phase is detailed in Algorithm 2.

A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
Table 6: Statistic ofHyGRAGusing Llama-3.1-8B-Instruct.
Metric PopQA QuALITY MultiHop-RAG MuSiQue HotpotQA
# of chunk nodes 32,157 1,518 1,658 29,898 66,559
# of entity nodes 271,776 33,440 36,884 224,846 510,457
# of community nodes 6,403 758 843 5,422 12,109
# of relations 379,564 32,938 32,968 314,145 758,380
# of chunk-chunk edge 162,273 3,690 16,466 48,264 199,675
# of chunk-entity edge 360,891 15,453 37,782 293,276 647,454
Table 7: Dataset Statistics
Dataset Tokens Questions of Chunks Avg. Tokens
PopQA 3,896,179 1,399 33,595(32,158) 121.16
QuALITY 1,521,377 4,609 265(1,518) 1084.25
MultiHop-RAG 1,424,248 2,556 609(1,658) 920.62
MuSiQue 3,134,496 3,000 29,898 104.84
HotpotQA 8,139,124 3,702 66,581(66,555) 122.29
Table 8: Hyperparameter settings used in our experiments.
Parameter Value
cleora_iterations 2
lsh_num_hyperplanes𝑘16
lsh_min_cluster_size𝑆 min 5
lsh_max_cluster_size𝑆 max 50
max_hierarchy_levels𝐿4
community_summary_length 300
shared_entity_threshold𝑙3
update_th𝑡𝑎𝑢 attach 0.65
HyGRAGseparates its operational logic into two distinct phases:
hierarchical index construction and dual-aware retrieval. Algorithm
1 delineates the creation of a hybrid graph that bridges unstructured
chunks with structured entity-relations, followed by a bottom-up
community abstraction process. Specifically, it employs LSH-based
clustering on Cleora embeddings and generates LLM summaries
creating emergent knowledge. Algorithm 2 details the retrieval
strategy, which synergizes context with relational triplets to pro-
vide a comprehensive context for the final generation. Specifically, it
performs bi-level retrieval across hierarchy layers while expanding
entity coverage through communities for comprehensive context.
C Experiment Details
We show the hybrid graph statistics, dataset statistics and Hy-
perparameter settings in Table 6, Table 7 and Table 8. Exploring
GNN [ 1,13,23–25] to learn more expressive representations over
heterogeneous structures is a future work. Regarding efficiency (on
Musique),HyGRAG’s offline stage (23,042s) is 21% faster than the
efficient GraphRAG representative (LightRAG). Online stage (10.7s)
is 7% faster than all graph-based baselines. During the process, it
uses 9.97GB memory and 2.50GB GPU (for embedding-model only).
Clustering takes 31s (GMM-clustering in RAPTOR took 10,965s).
Summarization stage requires 1 hour, about 15% of overall indexing
time. Results showHyGRAGis efficient in both offline and onlineAlgorithm 1HyGRAG Hierarchical Index Construction
Require:Corpus𝐷, Max layers𝐿, Split thresholds𝑆 𝑚𝑖𝑛,𝑆𝑚𝑎𝑥
Ensure:Hierarchical Index Tree𝑇 𝑖𝑛𝑑𝑒𝑥
1:Phase 1: Hybrid Graph Construction
2:𝐶←Chunking(𝐷)⊲Split documents into chunks
3:𝑉𝑐←𝐶;𝐸𝑐←∅
4:forpair(𝑐 𝑖,𝑐𝑗)in𝐶do
5:if|Entities(𝑐 𝑖)∩Entities(𝑐 𝑗)|>𝜖then
6:𝐸𝑐←𝐸𝑐∪{(𝑐𝑖,𝑐𝑗)}⊲Eq. 2
7:end if
8:end for
9:𝐺𝑒←ExtractTriplets(𝐶)⊲LLM extracts(ℎ,𝑟,𝑡)
10:𝐸𝑐𝑒←{(𝑒,𝑐)|𝑒∈𝑉𝑒,𝑐∈𝐶(𝑒)}⊲Eq. 6
11:𝐺ℎ𝑦𝑏𝑟𝑖𝑑←(𝑉𝑐∪𝑉𝑒,𝐸𝑐∪𝐸𝑒∪𝐸𝑐𝑒)
12:Phase 2: Hierarchical Clustering
13:𝑍←Cleora(𝐺 ℎ𝑦𝑏𝑟𝑖𝑑)⊲Structure-aware embeddings
14:L 0←𝑉(𝐺ℎ𝑦𝑏𝑟𝑖𝑑)
15:for𝑙=1to𝐿do
16:C𝑙←LSH_Clustering(L 𝑙−1,𝑍)⊲Eq. 8
17:L𝑙←∅
18:for𝑏𝑢𝑐𝑘𝑒𝑡∈C 𝑙do
19:𝐵←AdjustSize(𝑏𝑢𝑐𝑘𝑒𝑡,𝑆 𝑚𝑖𝑛,𝑆𝑚𝑎𝑥)
20:𝑡 𝐵←LLM𝑠𝑢𝑚𝑚({𝑣|𝑣∈𝐵})⊲Eq. 10
21:𝑠 𝐵←LM𝑒𝑚𝑏(𝑡𝐵)⊲Eq. 11
22:𝑍(𝑠 𝐵)←𝑠𝐵 ⊲Update embeddings for next layer
23:L 𝑙←L𝑙∪{𝑠𝐵}
24:Parent(𝐵)←𝑠 𝐵
25:end for
26:end for
27:return𝑇 𝑖𝑛𝑑𝑒𝑥={L 0,...,L𝐿}
phases. Regarding hallucinations in LLM-generated community
summaries, due to inherent limitations of LLM paradigm, using
LLMs inevitably introduces certain probability of hallucinations
(errors). However this issue can be mitigated as LLMs become more
powerful. Our framework allows replacing the LLM component,
and even lightweight models still improve performance. We manu-
ally evaluate 100 sampled summaries on Musique and found 17%
hallucination rate, mostly due to added external knowledge. After
switching to a stronger model (gpt-4o-mini) the rate dropped to 7%,
indicating improved summary quality.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates Haoyang Zhong et al.
Table 9: Overall QA results(Accuracy and Recall) on static RAG query datasets using Qwen3-8B. “OOT” denotes results that
could not be obtained within two days.
Method Factual Accuracy Reading Comprehension Multi-Hop Reasoning
PopQA QuALITY MultiHop-RAG MuSiQue HotpotQA
Accuracy Recall Accuracy Accuracy Recall Accuracy Recall Accuracy Recall
Zero-shot 25.23 9.83 34.78 47.10 27.33 8.13 22.38 30.01 36.62
CoT 30.52 14.75 36.06 60.02 31.37 10.50 27.97 32.90 41.15
VanillaRAG 63.83 35.00 59.28 61.93 37.40 23.60 35.26 51.54 55.15
RAPTOR 63.18 31.05 60.87 60.13 36.66 23.67 35.37 54.70 59.07
EraRAG OOT OOT61.7861.78 36.85 OOT OOT OOT OOT
L-LightRAG 71.05 44.37 58.41 55.52 38.25 33.10 47.52 OOT OOT
HippoRAG 63.97 33.49 56.65 61.89 33.54 24.90 38.88 OOT OOT
HippoRAG2 67.12 35.22 42.53 63.97 36.24 25.87 37.08 OOT OOT
L-GraphRAG 58.32 33.24 58.75 55.20 38.04 18.53 32.79 OOT OOT
HyGRAG73.84 46.1760.6773.87 (67.57) 45.09 (38.87) 36.67 49.33 72.18 72.67
Algorithm 2Context and Relation-Aware Retrieval
Require:Query𝑞, Index𝑇 𝑖𝑛𝑑𝑒𝑥 , Top-𝑘param𝑘
Ensure:Generated Answer𝐴
1:𝑞𝑒𝑚𝑏←LM𝑒𝑚𝑏(𝑞)⊲Encode query
2:Step 1: Context-Aware Retrieval
3:S𝑝𝑜𝑜𝑙←(Ð𝐿
𝑙=1L𝑙)∪𝑉𝑐
4:N𝑐𝑡𝑥←TopK(S 𝑝𝑜𝑜𝑙,𝑞𝑒𝑚𝑏,𝑘)
5:P𝑟𝑒𝑡←{𝑛∈N 𝑐𝑡𝑥|𝑛is Community}⊲Separate for prompt
6:𝐶𝑟𝑒𝑡←{𝑛∈N 𝑐𝑡𝑥|𝑛is Chunk}
7:𝐸𝑐𝑡𝑥
𝑟𝑒𝑡←TopK(𝑉𝑒,𝑞𝑒𝑚𝑏,𝑘)⊲Entities retrieved separately
8:Step 2: Relation-Aware Retrieval
9:𝐸𝑒𝑥𝑝𝑎𝑛𝑑←Ð
𝐶∈P 𝑟𝑒𝑡{𝑒|(𝑒,𝐶)∈𝐸𝑐𝑒}
10:𝐸𝑎𝑙𝑙←𝐸𝑐𝑡𝑥
𝑟𝑒𝑡∪𝐸𝑒𝑥𝑝𝑎𝑛𝑑 ⊲Integrate entities
11:𝑅𝑎𝑙𝑙←{(ℎ,𝑟,𝑡)|ℎ,𝑡∈𝐸 𝑎𝑙𝑙}⊲Extract connected triplets
12:𝑅𝑟𝑒𝑡←TopK(𝑅 𝑎𝑙𝑙,𝑞𝑒𝑚𝑏,𝑘)⊲Filter by similarity
13:Step 3: Generation
14:𝐶𝑡𝑥←Prompt(P 𝑟𝑒𝑡,𝐶𝑟𝑒𝑡,𝐸𝑐𝑡𝑥
𝑟𝑒𝑡,𝑅𝑟𝑒𝑡)⊲Structured Prompt
15:𝐴←LLM 𝑔𝑒𝑛(𝑞,𝐶𝑡𝑥)
16:return𝐴
Table 10: Performance comparison of different embedding
models under Vanilla andHyGRAGsettings.
Dense Retriever VanillaHyGRAG
Accuracy Recall Accuracy Recall
bge-m3 58.49 37.52 65.41 39.44
Qwen3-E-0.6B 57.98 37.45 65.49 39.39
text-embedding-v3 59.31 38.14 66.00 39.74
m-e5-large-instruct 58.76 37.83 63.65 39.06
D Extended Experiments
D.1 Robustness in Dense Retrievers
We experimented with a range of SOTA embedding models. As
shown in Table 10,HyGRAGconsistently outperforms the cor-
responding vanilla dense retrieval baseline across all embeddingmodels. Moreover, the performance ofHyGRAGimproves pro-
gressively as the quality of the underlying embedding model in-
creases, demonstrating its strong scalability and sensitivity to em-
bedding fidelity. In our experiments, we specifically evaluatedHy-
GRAGusing Qwen3-Embedding-0.6B[ 34], text-embedding-v3[ 34],
and multilingual-e5-large-instruct[ 28] as the underlying dense re-
trievers. This demonstrates thatHyGRAG’s retrieval framework
effectively amplifies embedding quality, creating synergistic gains
beyond vanilla dense retrieval approaches.
D.2 Robustness in Base LLM
We present the QA performance of various methods using the
base LLM Qwen3-8B, as summarized in Table 9. When employing
Qwen3-8B for both indexing and question answering,HyGRAG
consistently achieves competitive accuracy and recall across most
datasets, maintaining leading performance on MuSiQue and Hot-
potQA, with accuracy gains up to 21.36% over other methods.
For context-level methods, VanillaRAG, RAPTOR, and EraRAG
show slight improvements compared to their performance with the
previous base LLM. However, they exhibit noticeable drops on the
MuSiQue and HotpotQA datasets, indicating that their performance
is sensitive to changes in the base LLM. Similarly, the relation-level
method HippoRAG2 also shows performance declines on these
datasets, and its results on QuALITY remain suboptimal.
In contrast,HyGRAGmaintains stable and superior perfor-
mance across all tasks, demonstrating robustness and adaptability
to different LLM architectures. This suggests thatHyGRAG’s
synergistic integration of relation- and context-level information
provides resilience against base model changes, while other ap-
proaches are more sensitive to variations in the underlying LLM.
E Prompts Used forHyGRAG
The Answer Generation Prompt is designed to synthesize a com-
prehensive and evidence-grounded answer to a user’s query from a
structured, multi-source context. The process begins by presenting
the model with a rich contextual payload retrieved from a hierar-
chical knowledge system, which includes: hierarchical community

A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates
summaries with similarity scores, a ranked list of the most rele-
vant entities, key entity relationships structured as triplets, and the
content of the most relevant source documents. Subsequently, the
prompt provides a set of explicit operational rules to constrain the
generation process. The framework mandates that the model must
report any informational gaps by stating "Insufficient information"
and is forbidden from fabricating information not present in the
context, ensuring a high degree of fidelity to the source material.
The Community Summarization Prompt outlines a framework
for distilling a collection of content, referred to as a knowledge com-
munity, into a structured and semantically dense summary. The
prompt’s primary objective is to generate a summary specificallytailored for the downstream task of creating high-quality seman-
tic embeddings. It guides the model by defining a clear, four-part
structure for the output, requiring the summary to comprehensively
cover: 1) key themes and topics, 2) important entities and their roles,
3) relationships and connections between these entities, and 4) the
overall context and significance of the information. By specifying
both a word limit and the explicit need for rich semantic content,
the prompt ensures the generation of a concise yet information-rich
text that captures the core essence of the knowledge community.
The detailed prompt are provided in the code repository https:
//github.com/zjunet/HyGRAG.