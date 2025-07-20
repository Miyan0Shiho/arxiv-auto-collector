# Extracting Document Relations from Search Corpus by Marginalizing over User Queries

**Authors**: Yuki Iwamoto, Kaoru Tsunoda, Ken Kaneiwa

**Published**: 2025-07-14 18:47:13

**PDF URL**: [http://arxiv.org/pdf/2507.10726v1](http://arxiv.org/pdf/2507.10726v1)

## Abstract
Understanding relationships between documents in large-scale corpora is
essential for knowledge discovery and information organization. However,
existing approaches rely heavily on manual annotation or predefined
relationship taxonomies. We propose EDR-MQ (Extracting Document Relations by
Marginalizing over User Queries), a novel framework that discovers document
relationships through query marginalization. EDR-MQ is based on the insight
that strongly related documents often co-occur in results across diverse user
queries, enabling us to estimate joint probabilities between document pairs by
marginalizing over a collection of queries. To enable this query
marginalization approach, we develop Multiply Conditioned Retrieval-Augmented
Generation (MC-RAG), which employs conditional retrieval where subsequent
document retrievals depend on previously retrieved content. By observing
co-occurrence patterns across diverse queries, EDR-MQ estimates joint
probabilities between document pairs without requiring labeled training data or
predefined taxonomies. Experimental results show that our query marginalization
approach successfully identifies meaningful document relationships, revealing
topical clusters, evidence chains, and cross-domain connections that are not
apparent through traditional similarity-based methods. Our query-driven
framework offers a practical approach to document organization that adapts to
different user perspectives and information needs.

## Full Text


<!-- PDF content starts -->

Extracting Document Relations from Search Corpus by
Marginalizing over User Queries
Yuki Iwamoto
iwamoto@sw.cei.uec.ac.jpKaoru Tsunoda
tsunoda@uec.ac.jpKen Kaneiwa
kaneiwa@uec.ac.jp
The University of Electro-Communications
Chofu, Tokyo, Japan
Abstract
Understanding relationships between documents in large-
scale corpora is essential for knowledge discovery and in-
formation organization. However, existing approaches rely
heavily on manual annotation or predefined relationship
taxonomies. We propose EDR-MQ (Extracting Document
Relations by Marginalizing over User Queries), a novel
framework that discovers document relationships through
query marginalization. EDR-MQ is based on the insight
that strongly related documents often co-occur in results
across diverse user queries, enabling us to estimate joint
probabilities between document pairs by marginalizing over
a collection of queries. To enable this query marginaliza-
tion approach, we develop Multiply Conditioned Retrieval-
Augmented Generation (MC-RAG), which employs condi-
tional retrieval where subsequent document retrievals de-
pend on previously retrieved content. By observing co-
occurrence patterns across diverse queries, EDR-MQ es-
timates joint probabilities between document pairs without
requiring labeled training data or predefined taxonomies.
Experimental results show that our query marginalization
approach successfully identifies meaningful document rela-
tionships, revealing topical clusters, evidence chains, and
cross-domain connections that are not apparent through
traditional similarity-based methods. Our query-driven
framework offers a practical approach to document orga-
nization that adapts to different user perspectives and in-
formation needs.
1. Introduction
The exponential growth of digital documents across various
domains has created an urgent need for automated meth-
ods to understand and organize large-scale document col-
lections [4]. While traditional information retrieval systems
excel at finding relevant documents for specific queries,
Figure 1. Overview of the EDR-MQ framework for extracting
document relationships through user query marginalization
they fall short in revealing the underlying relational struc-
ture that connects documents within a corpus [3, 13]. Un-
derstanding these document relationships is crucial for ap-
plications ranging from knowledge discovery and semantic
search to automated literature review and content recom-
mendation systems [2, 4].
Recent advances in Retrieval-Augmented Generation
(RAG) have demonstrated remarkable success in leveraging
external knowledge for improved text generation [9]. How-
ever, while RAG systems effectively retrieve relevant docu-
ments for generation, the relationships between documents
remain opaque to users. This limitation makes it difficult
for users to verify facts, understand the broader context, or
reuse the knowledge for other purposes, as they cannot see
how different pieces of information connect to each other
within the corpus.
Consider a scenario where teams in large organizations
need to navigate vast internal document repositories con-
taining policies, reports, procedures, and project documen-
tation. While current RAG-based systems can help employ-
ees retrieve relevant documents, the relationships between
different pieces of organizational knowledge remain hid-
1arXiv:2507.10726v1  [cs.IR]  14 Jul 2025

den, making it difficult for teams to understand how poli-
cies connect to procedures, how different projects relate
to each other, or how decisions in one department might
impact others. This lack of document relationship visibil-
ity hampers collaboration, leads to inconsistent decision-
making across teams, and prevents organizations from fully
leveraging their accumulated knowledge assets.
In this work, we introduce EDR-MQ (Extracting Doc-
ument Relations by Marginalizing over User Queries), a
novel framework that addresses these limitations by dis-
covering document relationships through query marginal-
ization. Our key insight is that by observing how docu-
ments co-occur across diverse user queries in retrieval sys-
tems, we can infer their underlying relationships without re-
quiring explicit supervision or predefined relationship tax-
onomies. To enable this approach, we develop a Multiply
Conditioned Retrieval-Augmented Generation (MC-RAG)
mechanism where subsequent document retrievals are con-
ditioned on previously retrieved content.
EDR-MQ operates on the principle that documents with
strong relationships will frequently co-occur when retrieved
for semantically related queries. By marginalizing over a di-
verse collection of user queries, we can estimate joint prob-
abilities between document pairs, effectively transforming
the retrieval process into a relationship discovery mecha-
nism. This probabilistic framework enables us to construct
relationship matrices that capture the strength of associa-
tions between documents, facilitating downstream applica-
tions such as knowledge graph construction and semantic
clustering.
The main contributions of this paper are:
• We propose EDR-MQ, a novel framework for extract-
ing document relationships by marginalizing over user
queries without requiring labeled training data or prede-
fined taxonomies.
• We develop MC-RAG, a conditional retrieval mechanism
that enables the capture of inter-document dependencies
by conditioning subsequent retrievals on previously re-
trieved documents.
• We demonstrate the effectiveness of EDR-MQ through
comprehensive experiments on scientific literature, show-
ing that our method can discover meaningful document
relationships and reveal corpus structure.
Our experimental results on the SciFact dataset demon-
strate that the proposed method successfully identifies
meaningful relationships between scientific claims and
evidence, outperforming traditional similarity-based ap-
proaches in discovering latent document connections.2. Related Works
2.1. Retrieval Augmented Generation
Retrieval-Augmented Generation (RAG) has emerged as a
powerful paradigm that combines the strengths of paramet-
ric and non-parametric approaches to knowledge-intensive
natural language processing tasks [9]. The core idea
of RAG is to augment language models with the ability
to retrieve relevant information from external knowledge
sources during generation.
The standard RAG architecture consists of two main
components: a retriever and a generator. The retriever, typ-
ically implemented using dense passage retrieval methods,
identifies relevant documents from a large corpus given a
query. The generator, usually a pre-trained language model
such as BART [8] or T5 [10], then conditions its generation
on both the input query and the retrieved passages. For-
mally, RAG computes the probability of generating output
ygiven input xas:
p(y|x) =X
zp(z|x)p(y|x, z) (1)
where zrepresents the retrieved passages, p(z|x)is the re-
trieval probability, and p(y|x, z)is the generation probabil-
ity.
Understanding the relationships between retrieved pas-
sages in RAG systems remains challenging. RagViz [12]
addresses this limitation by visualizing attention patterns
between queries and retrieved passages. However, RagViz
focuses on understanding individual query-passage interac-
tions rather than discovering broader relationships between
documents across multiple queries.
2.2. Dense Passage Retrieval and ColBERT
Dense Passage Retrieval (DPR) [6] revolutionized infor-
mation retrieval by replacing traditional sparse retrieval
methods (such as BM25) with dense vector representations
learned through deep learning. DPR encodes queries and
passages into dense vectors using dual-encoder architec-
tures, typically based on BERT [5], and performs retrieval
by computing similarity in the embedding space.
While DPR and similar bi-encoder approaches achieve
strong retrieval performance, they suffer from the represen-
tation bottleneck problem: all information about a passage
must be compressed into a single dense vector. This limita-
tion led to the development of late interaction models such
as ColBERT [7].
ColBERT (Contextualized Late Interaction over BERT)
addresses the representation bottleneck by postponing the
interaction between query and document representations
until after encoding. Instead of producing single vectors,
ColBERT generates a sequence of contextualized embed-
dings for each token in both queries and documents. The
2

similarity between a query qand document dis computed
as:
Score (q, d) =X
i∈|q|max
j∈|d|E(i)
q·E(j)
d(2)
where E(i)
qandE(j)
dare the contextualized embeddings of
thei-th query token and j-th document token, respectively.
This late interaction mechanism allows ColBERT to cap-
ture fine-grained matching signals between query and doc-
ument tokens. ColBERT has demonstrated superior perfor-
mance across various retrieval benchmarks and has been
adopted in numerous applications requiring high-quality
document retrieval.
The choice of ColBERT as our retrieval backbone
is motivated by its ability to capture nuanced relation-
ships between queries and documents, which is crucial for
our conditional retrieval mechanism where the second re-
triever must understand the relationship between the origi-
nal query, the first retrieved document, and candidate docu-
ments for the second retrieval step.
Limitations of Existing Approaches: Most existing
document relation extraction methods face several limita-
tions: (1) They require predefined relationship taxonomies
or labeled training data; (2) They operate on static docu-
ment representations without considering query context; (3)
They cannot adapt to different user perspectives or informa-
tion needs;
Our approach addresses these limitations by leveraging
the query marginalization process in MC-RAG to discover
document relationships in an unsupervised manner, without
requiring predefined relationship types or labeled training
data. The conditional retrieval mechanism allows us to cap-
ture context-dependent relationships that vary based on user
queries and information needs.
3. Question-based Relation Extraction Frame-
work
In this section, we introduce our proposed framework,
which extracts document relations by marginalizing over
user queries. Our framework extracts document relations
using MC-RAG’s retrievers. MC-RAG is a RAG that has
multiple retrievers to retrieve evidence sentences and then
generates answer text. Firstly, we introduce MC-RAG in
the following section. Next, we describe how to extract doc-
ument relations using MC-RAG by marginalizing over user
queries.
3.1. Multiply Conditioned RAG
MC-RAG is a RAG model that takes a user input xto re-
trieve multiple text documents zi, . . . , z k, which are then
used as additional context for generating the answer sen-
tence y. Without loss of generality, we describe our model
withk= 2. MC-RAG has kretrievers ( k= 2):pη1(zi|x)andpη2(zj|zi, x), which return distributions over text pas-
sages given input xand(zi, x), respectively.
We employ ColBERT as the encoder for both retriev-
ers, which computes fine-grained similarity scores between
query and document representations. ColBERT encodes
queries and documents into contextualized embeddings at
the token level, enabling efficient and accurate retrieval
through late interaction mechanisms. The similarity com-
putation follows ColBERT’s approach, where the similarity
between query qand document dis computed as:
sim(q, d) =X
i∈|q|max
j∈|d|Ei
q·Ej
d(3)
where Ei
qandEj
dare the contextualized embeddings of the
i-th query token and j-th document token, respectively.
The key innovation of MC-RAG lies in its conditional
retrieval mechanism pη2(zj|zi, x). Unlike standard RAG
models that perform independent retrieval, the second re-
triever conditions its search on both the user input xand the
previously retrieved passage zi. This conditional retrieval
is achieved by concatenating the embeddings of xandzi
as the query representation for the second retriever. Specif-
ically, the second retriever computes similarities using the
combined context [zi;x]where [; ]denotes concatenation.
This conditional mechanism is crucial for enabling the
computation of joint probabilities p(zi, zj)as described in
the subsequent section. A single retriever cannot capture
such dependencies between retrieved passages, as it oper-
ates independently for each retrieval step. The conditional
structure allows MC-RAG to model relationships between
documents and extract structured information from the cor-
pus by marginalizing over user queries.
Similar to standard RAG models, MC-RAG has a gener-
atorpθ(y|zi, zj, x)that produces the output yconditioned
on both the input xand the retrieved passages ziandzj.
The final output distribution is obtained by marginalizing
over all possible combinations of retrieved passages:
p(y|x) =X
ziX
zjpη1(zi|x)pη2(zj|zi, x)pθ(y|x, zi, zj)
(4)
3.2. Extracting Document Relations by Marginaliz-
ing over User Queries
In this section, we describe how to extract document rela-
tions by marginalizing over user queries. We calculate the
joint probability of the retrieved passages with the following
equation:
p(zi, zj) =X
xpη1(zi|x)pη2(zj|zi, x)p(x) (5)
Therefore, our method leverages the diversity of user
queries to estimate the underlying relationships between
3

Figure 2. Model Architecture
documents in the corpus. The more diverse and compre-
hensive the set of user queries, the more accurate the joint
probability estimation becomes, as it captures different as-
pects and contexts in which documents ziandzjare rele-
vant together.
The key insight is that by marginalizing over a large col-
lection of user queries, we can discover latent relationships
between documents that may not be apparent from individ-
ual queries alone. Each query xprovides a different per-
spective on how documents relate to each other, and the
aggregation across multiple queries reveals stable patterns
of document co-occurrence and dependency. This approach
effectively transforms the retrieval process into a structure
discovery mechanism, where the conditional retrieval pat-
terns learned by MC-RAG expose the underlying semantic
relationships in the corpus.
In practice, we collect a diverse set of queries X=
{x1, x2, . . . , x N}and approximate the marginal distribu-
tion as:
p(zi, zj)≈1
NNX
n=1pη1(zi|xn)pη2(zj|zi, xn) (6)
where we assume a uniform prior p(x) =1
Nover the query
set. The resulting joint probabilities p(zi, zj)form a re-
lationship matrix that captures the strength of associations
between different document pairs in the corpus.
3.2.1. Algorithm for Document Relation Extraction
We now present the detailed algorithm for extracting doc-
ument relations using MC-RAG. Algorithm 1 outlines the
complete process from query collection to relationship ma-
trix construction.
The algorithm operates in three main phases:Algorithm 1 Document Relation Extraction via Query
Marginalization
Require: Document corpus D={d1, d2, . . . , d M}, Query
setX={x1, x2, . . . , x N}, Top-k retrieval parameter k
Ensure: Relationship matrix R∈RM×M
1:Initialize relationship matrix Rwith zeros
2:Precompute document embeddings using ColBERT en-
coder
3:foreach query xn∈ X do
4:Z1←Retrieve top- kdocuments using first retriever
pη1(zi|xn)
5: foreach retrieved document zi∈ Z 1do
6: Construct conditional query qcond= [zi;xn]
7: Z2←Retrieve top- kdocuments using second
retriever pη2(zj|zi, xn)
8: foreach retrieved document zj∈ Z 2do
9: p1←pη1(zi|xn) ▷First retriever
probability
10: p2←pη2(zj|zi, xn) ▷Second retriever
probability
11: R[i, j]←R[i, j] +p1·p2▷Accumulate
joint probability
12: end for
13: end for
14:end for
15:return Relationship matrix R
Preprocessing Phase (Lines 1-2): We initialize the rela-
tionship matrix and precompute document embeddings us-
ing ColBERT to enable efficient similarity computation dur-
ing retrieval.
Query Processing Phase (Lines 3-12): For each query
in the collection, we perform the two-stage retrieval pro-
4

cess. The first retriever identifies relevant documents based
on the query alone, while the second retriever finds docu-
ments that are relevant given both the query and the previ-
ously retrieved document. The joint probabilities are accu-
mulated in the relationship matrix, effectively marginalizing
over the query distribution.
The computational complexity of this algorithm is O(N·
k2·C), where Nis the number of queries, kis the number
of retrieved documents per stage, and Cis the cost of sim-
ilarity computation in ColBERT. The space complexity is
O(M2)for storing the relationship matrix, where Mis the
total number of documents in the corpus.
4. Experiments
In this section, we demonstrate the effectiveness of our pro-
posed EDR-MQ framework through experiments. We focus
on qualitative analyses and visualization results that show-
case the method’s ability to extract meaningful document
relationships and reveal corpus structure as the diversity of
user queries increases.
4.1. Experimental Setup
We evaluate our EDR-MQ framework on the SciFact
dataset [11], which contains 1,409 scientific claims paired
with 5,183 abstracts. For our experiments, we treat each
claim as a user query, and paper abstracts as a corpus.
For visualization, we employed Gephi with the ForceAt-
las 2 algorithm for graph layout, using the parameters sum-
marized in Table 1.
4.2. Effect of Query Diversity on Relationship Dis-
covery
A fundamental hypothesis of our approach is that increasing
the diversity and number of user queries leads to more com-
prehensive document relationship extraction. To validate
this hypothesis, we conducted experiments using different
numbers of queries on the SciFact dataset and analyzed the
resulting relationship networks.
Figure 3 illustrates the effect of query diversity on rela-
tionship discovery. When using only 30 queries (Figure 3a),
which represents 1/10 of our full query set, the extracted re-
lationship network exhibits numerous small, isolated clus-
ters. While the method successfully identifies local relation-
ships within these clusters, the limited query diversity fails
to reveal connections between different clusters, resulting
in a fragmented understanding of the corpus structure.
In contrast, when the full set of 300 queries is employed
(Figure 3b), the relationship network becomes significantly
more connected and comprehensive. The previously iso-
lated clusters are now linked through bridge documents that
connect different parts of the corpus. This demonstrates
that our query marginalization approach uncovers more la-
tent relationships as the number of queries increases, reveal-ing connections that are not apparent when using limited
queries.
The key insights from this experiment are:
•Query Diversity is Critical : More diverse queries enable
the discovery of relationships across different parts of the
corpus, preventing the formation of isolated clusters.
•Bridge Document Discovery : With sufficient query di-
versity, documents that serve as bridges between different
clusters become apparent, revealing the interconnected
nature of the corpus.
•Scalable Relationship Extraction : The method grace-
fully scales with query diversity, progressively revealing
corpus structure without requiring additional supervision.
This experiment supports our theoretical framework
where joint probabilities p(zi, zj)become more accurate as
we marginalize over a larger and more diverse set of user
queries, leading to better estimation of document relation-
ships.
4.3. Network Interpretability and Visualization
Analysis
To demonstrate the advantages of our EDR-MQ frame-
work, we compare it with traditional document similarity
approaches through comprehensive network analysis and
visualization.
Figure 5 shows the document relationship network con-
structed using TF-IDF similarity, where each document is
connected to its top-25 most similar documents based on
term frequency-inverse document frequency scores. This
top-25 setting corresponds to the same number of document
pairs that our EDR-MQ method considers when performing
5 retrievals in the first stage followed by 5 retrievals in the
second stage (5×5=25).
The TF-IDF-based network exhibits several problem-
atic characteristics that highlight the limitations of tradi-
tional similarity measures. The network is densely inter-
connected, creating a tangled web that obscures meaningful
relationship patterns. This excessive complexity makes it
extremely difficult to identify coherent document clusters
or understand the semantic basis of connections. Moreover,
most documents have similar numbers of connections, fail-
ing to distinguish between central hub documents and pe-
ripheral nodes. Unlike our EDR-MQ results, the TF-IDF
network lacks a clear community structure or hierarchical
organization, presenting an absence of interpretable struc-
ture.
To further analyze the structural differences between tra-
ditional similarity measures and our EDR-MQ approach,
we applied edge bundling visualization to both networks.
Edge bundling [14] is a graph visualization technique that
groups similar edges together into bundles, reducing vi-
sual clutter and revealing high-level connectivity patterns in
complex networks. This technique is particularly useful for
5

(a) Document relationships with 30 queries
 (b) Document relationships with 300 queries
Figure 3. Effect of query diversity on document relationship extraction. (a) With only 30 queries (1/10 of the full set), the method creates
numerous small, isolated clusters with limited inter-cluster connections. (b) With 300 queries, the relationships become more comprehen-
sive, connecting previously isolated clusters and revealing the overall corpus structure through richer inter-document connections.
Figure 4. Concrete example of extracted document relationships across different scientific categories. The figure shows how our MC-RAG
framework discovers meaningful connections between documents from different domains: Microbiology, Cell Biology, Molecular Biology
(two instances), and Immunology. Despite belonging to different categories, the documents are connected through shared concepts such as
neutrophil extracellular traps (NETs), bacterial immunity mechanisms, and cellular processes. This demonstrates the method’s ability to
identify cross-domain relationships that would not be apparent through traditional category-based organization.
6

Figure 5. Document relationship network using TF-IDF simi-
larity with top-25 connections per document. The network ex-
hibits excessive complexity with dense interconnections that ob-
scure meaningful patterns, making it difficult to interpret docu-
ment relationships and identify coherent clusters.
understanding the overall flow and structure of document
relationships.
Figure 6 presents a compelling comparison between TF-
IDF similarity networks and our EDR-MQ approach using
edge bundling visualization. Documents are color-coded by
their categorical labels and positioned using the ForceAtlas
2 layout algorithm. The edge bundling technique groups re-
lated connections together, making it easier to identify ma-
jor relationship pathways and structural patterns.
The TF-IDF-based network (Figure 6a) exhibits scat-
tered edge patterns with poor bundling characteristics. The
edges appear dispersed across the visualization space with
little coherent grouping, making it difficult to discern mean-
ingful relationship trends or identify major connection path-
ways between document categories. This scattered pattern
reflects the lexical nature of TF-IDF similarity, which often
produces connections based on surface-level term overlap
rather than deeper semantic relationships.
In contrast, the EDR-MQ network (Figure 6b) demon-
strates clear and well-defined edge bundling patterns. The
relationships form coherent bundles that connect related
document clusters, creating interpretable pathways that re-
flect meaningful semantic connections. These bundled
edges reveal the underlying structure of how different doc-
ument categories relate to each other through query-driven
discovery, making the overall network organization muchmore comprehensible.
This comprehensive comparison demonstrates that our
EDR-MQ approach produces more structured and inter-
pretable document relationships compared to traditional
similarity measures, facilitating better understanding of cor-
pus structure and enabling more effective navigation of doc-
ument collections.
4.4. Cross-Domain Relationship Discovery
An important characteristic of our approach is its ability
to discover relationships between documents that span dif-
ferent categorical boundaries based on the diversity of user
queries.
Figure 4 presents a concrete example of how our EDR-
MQ framework extracts meaningful relationships between
scientific documents from diverse categories. The visual-
ization shows five documents from four different biological
domains:
•Microbiology : Focuses on neutrophil extracellular trap
formation in response to Staphylococcus aureus
•Cell Biology : Discusses neutrophil extracellular chro-
matin traps and their connection to autoimmune re-
sponses
•Molecular Biology (Document 1): Examines PAD4’s
role in antibacterial immunity and neutrophil extracellu-
lar traps
•Molecular Biology (Document 2): Investigates viable
neutrophils and mitochondrial DNA release in NET for-
mation
•Immunology : Studies neutrophil extracellular traps and
chromatin granule function
Despite being classified into different categories, our
method successfully identifies the underlying conceptual
connections between these documents. The extracted re-
lationships reveal a coherent research narrative centered
around neutrophil extracellular traps (NETs), bacterial im-
munity mechanisms, and cellular processes involved in im-
mune defense.
This cross-domain relationship discovery capability is
further validated by the edge bundling analysis shown in
Figure 6. The clear bundling patterns in our EDR-MQ net-
work (Figure 6b) demonstrate how documents from differ-
ent categories are connected through meaningful semantic
pathways, whereas the TF-IDF approach fails to reveal such
coherent cross-domain connections. The superior network
organization in our approach enables the identification of
interdisciplinary relationships that would be missed by tra-
ditional category-based organization methods.
This example demonstrates that our query marginaliza-
tion approach can transcend traditional categorical bound-
aries to reveal the true conceptual structure underlying sci-
entific literature. Such cross-domain relationship discovery
is particularly valuable for interdisciplinary research, where
7

(a) TF-IDF similarity network with edge bundling
 (b) EDR-MQ relationship network with edge bundling
Figure 6. Comparison of document relationship networks using edge bundling visualization. Documents are color-coded by category and
positioned using ForceAtlas 2 layout. (a) TF-IDF similarity network shows scattered edge patterns with poor bundling, making overall
trends difficult to interpret. (b) EDR-MQ network exhibits clear edge bundling patterns that reveal coherent relationship structures and
facilitate interpretation.
Table 1. ForceAtlas 2 Layout Parameters
Parameter Value
Tolerance (speed) 1.0
Approximate Repulsion Enabled
Approximation 1.2
Scaling 2.0
Stronger Gravity Disabled
Gravity 1.0
Dissuade Hubs Enabled
LinLog mode Enabled
Prevent Overlap Enabled
Edge Weight Influence 1.0
Normalize edge weights Enabled
Inverted edge weights Enabled
important insights often emerge from connections between
seemingly disparate fields.
5. Conclusion
We introduced EDR-MQ, a novel framework for extracting
document relationships through query marginalization us-
ing Multiply Conditioned RAG (MC-RAG). Our approach
discovers latent relationships without requiring manual an-
notation or predefined taxonomies by marginalizing over di-
verse user queries.
The key innovation lies in the conditional retrieval mech-
anism, where subsequent retrievals depend on previously
retrieved content. This enables the construction of relation-ship matrices that reveal corpus structure through query-
driven co-occurrence patterns.
Experimental results on the SciFact dataset demonstrate
that query diversity is critical for comprehensive relation-
ship extraction, and our method successfully identifies
cross-domain relationships that transcend traditional cate-
gorical boundaries. The unsupervised nature of our ap-
proach makes it particularly valuable for domains where
manual relationship annotation is prohibitively expensive.
In the future, we plan to explore the following directions:
First, end-to-end training of the entire MC-RAG framework
could potentially improve the quality of extracted relation-
ships by jointly optimizing the retrieval and relationship ex-
traction objectives. Second, incorporating advanced edge
bundling techniques that provide localized zoom views [1]
could enable better understanding of both local and global
document relationships. Finally, comprehensive evalua-
tion across diverse datasets and comparison with additional
baseline methods would strengthen the validation of our ap-
proach’s generalizability.
References
[1] KEIICHI AKIYAMA, HIDEYUKI FUJITA, TADASHI
OMORI, and TAKAHIKO SHINTANI. Focus+ context edge
bundling for network visualization. Journal of Information
Processing Society of Japan , 65(3):667–676, 2024. 8
[2] Muhammad Arslan, Saba Munawar, and Christophe Cruz.
Business insights using rag–llms: a review and case study.
Journal of Decision Systems , pages 1–30, 2024. 1
[3] Boqi Chen, Kua Chen, Yujing Yang, Afshin Amini,
8

Bharat Saxena, Cecilia Ch ´avez-Garc ´ıa, Majid Babaei, Amir
Feizpour, and D ´aniel Varr ´o. Towards improving the explain-
ability of text-based information retrieval with knowledge
graphs. arXiv preprint arXiv:2301.06974 , 2023. 1
[4] Lei Cui, Yiheng Xu, Tengchao Lv, and Furu Wei. Doc-
ument AI: benchmarks, models and applications. CoRR ,
abs/2111.08609, 2021. 1
[5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. BERT: pre-training of deep bidirectional trans-
formers for language understanding. In Proceedings of the
2019 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Lan-
guage Technologies, NAACL-HLT 2019, Minneapolis, MN,
USA, June 2-7, 2019, Volume 1 (Long and Short Papers) ,
pages 4171–4186. Association for Computational Linguis-
tics, 2019. 2
[6] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-
tau Yih. Dense passage retrieval for open-domain question
answering. In Proceedings of the 2020 Conference on Em-
pirical Methods in Natural Language Processing, EMNLP
2020, Online, November 16-20, 2020 , pages 6769–6781. As-
sociation for Computational Linguistics, 2020. 2
[7] Omar Khattab and Matei Zaharia. Colbert: Efficient and
effective passage search via contextualized late interaction
over BERT. In Proceedings of the 43rd International ACM
SIGIR conference on research and development in Informa-
tion Retrieval, SIGIR 2020, Virtual Event, China, July 25-30,
2020 , pages 39–48. ACM, 2020. 2
[8] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvinine-
jad, Abdelrahman Mohamed, Omer Levy, Veselin Stoy-
anov, and Luke Zettlemoyer. BART: denoising sequence-to-
sequence pre-training for natural language generation, trans-
lation, and comprehension. In Proceedings of the 58th An-
nual Meeting of the Association for Computational Linguis-
tics, ACL 2020, Online, July 5-10, 2020 , pages 7871–7880.
Association for Computational Linguistics, 2020. 2
[9] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, Sebas-
tian Riedel, and Douwe Kiela. Retrieval-augmented genera-
tion for knowledge-intensive NLP tasks. In Advances in Neu-
ral Information Processing Systems 33: Annual Conference
on Neural Information Processing Systems 2020, NeurIPS
2020, December 6-12, 2020, virtual , 2020. 1, 2
[10] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee,
Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and
Peter J. Liu. Exploring the limits of transfer learning with
a unified text-to-text transformer. J. Mach. Learn. Res. , 21:
140:1–140:67, 2020. 2
[11] David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang,
Madeleine van Zuylen, Arman Cohan, and Hannaneh Ha-
jishirzi. Fact or fiction: Verifying scientific claims. In
Proceedings of the 2020 Conference on Empirical Meth-
ods in Natural Language Processing, EMNLP 2020, Online,
November 16-20, 2020 , pages 7534–7550. Association for
Computational Linguistics, 2020. 5[12] Tevin Wang, Jingyuan He, and Chenyan Xiong. Ragviz: Di-
agnose and visualize retrieval-augmented generation. CoRR ,
abs/2411.01751, 2024. 2
[13] Xueli Yu, Weizhi Xu, Zeyu Cui, Shu Wu, and Liang Wang.
Graph-based hierarchical relevance matching signals for ad-
hoc retrieval. In Proceedings of the Web Conference 2021 ,
pages 778–787, 2021. 1
[14] Hong Zhou, Panpan Xu, Xiaoru Yuan, and Huamin Qu. Edge
bundling in information visualization. Tsinghua Science and
Technology , 18(2):145–156, 2013. 5
9