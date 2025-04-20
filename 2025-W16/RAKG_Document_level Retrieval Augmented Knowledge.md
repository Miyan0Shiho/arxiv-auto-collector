# RAKG:Document-level Retrieval Augmented Knowledge Graph Construction

**Authors**: Hairong Zhang, Jiaheng Si, Guohang Yan, Boyuan Qi, Pinlong Cai, Song Mao, Ding Wang, Botian Shi

**Published**: 2025-04-14 02:47:23

**PDF URL**: [http://arxiv.org/pdf/2504.09823v1](http://arxiv.org/pdf/2504.09823v1)

## Abstract
With the rise of knowledge graph based retrieval-augmented generation (RAG)
techniques such as GraphRAG and Pike-RAG, the role of knowledge graphs in
enhancing the reasoning capabilities of large language models (LLMs) has become
increasingly prominent. However, traditional Knowledge Graph Construction (KGC)
methods face challenges like complex entity disambiguation, rigid schema
definition, and insufficient cross-document knowledge integration. This paper
focuses on the task of automatic document-level knowledge graph construction.
It proposes the Document-level Retrieval Augmented Knowledge Graph Construction
(RAKG) framework. RAKG extracts pre-entities from text chunks and utilizes
these pre-entities as queries for RAG, effectively addressing the issue of
long-context forgetting in LLMs and reducing the complexity of Coreference
Resolution. In contrast to conventional KGC methods, RAKG more effectively
captures global information and the interconnections among disparate nodes,
thereby enhancing the overall performance of the model. Additionally, we
transfer the RAG evaluation framework to the KGC field and filter and evaluate
the generated knowledge graphs, thereby avoiding incorrectly generated entities
and relationships caused by hallucinations in LLMs. We further developed the
MINE dataset by constructing standard knowledge graphs for each article and
experimentally validated the performance of RAKG. The results show that RAKG
achieves an accuracy of 95.91 % on the MINE dataset, a 6.2 % point improvement
over the current best baseline, GraphRAG (89.71 %). The code is available at
https://github.com/LMMApplication/RAKG.

## Full Text


<!-- PDF content starts -->

RAKG:Document-level Retrieval Augmented Knowledge Graph
Construction
Hairong Zhang1,2, Jiaheng Si1,3, Guohang Yan†1, Boyuan Qi4, Pinlong Cai1,
Song Mao1, Ding Wang1, Botian Shi1
Abstract — With the rise of knowledge graph based retrieval-
augmented generation (RAG) techniques such as GraphRAG
and Pike-RAG, the role of knowledge graphs in enhancing the
reasoning capabilities of large language models (LLMs) has
become increasingly prominent. However, traditional Knowl-
edge Graph Construction (KGC) methods face challenges like
complex entity disambiguation, rigid schema definition, and
insufficient cross-document knowledge integration. This paper
focuses on the task of automatic document-level knowledge
graph construction. It proposes the Document-level Retrieval
Augmented Knowledge Graph Construction (RAKG) frame-
work. RAKG extracts pre-entities from text chunks and utilizes
these pre-entities as queries for RAG, effectively addressing
the issue of long-context forgetting in LLMs and reducing the
complexity of Coreference Resolution. In contrast to conven-
tional KGC methods, RAKG more effectively captures global
information and the interconnections among disparate nodes,
thereby enhancing the overall performance of the model. Addi-
tionally, we transfer the RAG evaluation framework to the KGC
field and filter and evaluate the generated knowledge graphs,
thereby avoiding incorrectly generated entities and relationships
caused by hallucinations in LLMs. We further developed the
MINE dataset by constructing standard knowledge graphs for
each article and experimentally validated the performance of
RAKG. The results show that RAKG achieves an accuracy of
95.91% on the MINE dataset, a 6.2 % point improvement over
the current best baseline, GraphRAG (89.71%). The code is
available at https://github.com/LMMApplication/RAKG.
I. I NTRODUCTION
With the development of LLMs [1], their remarkable
capabilities have been increasingly evident, thereby offering
novel insights for further innovation across diverse domains.
However, LLMs also have limitations. For example, LLMs
cannot acquire knowledge beyond their training data and
often omit crucial information when processing long texts.
RAG technology [2], which leverages vector retrieval, has,
to some extent, addressed the issues of delayed knowledge
training and limited context length and has played a signif-
icant role in multiple fields. As RAG technology continues
to evolve, the success of GraphRAG [3] and Pike-RAG [4]
has further proven the importance of knowledge graphs.
Therefore, establishing a comprehensive and high-quality
knowledge graph is essential.
In the field of KGC, traditional methods are no longer
sufficient. Rule-based approaches [5] are costly, inflexible,
†Corresponding author. yanguohang@pjlab.org.cn
1Shanghai Artificial Intelligence Laboratory, China.
2School of Statistics and Data Science, Nankai University, Tianjin, China.
3Faculty of Computing, Harbin Institute of Technology, China.
4Computer School, Beijing Information Science and Technology Univer-
sity, China.and struggle to adapt to new domains. Machine learning
methods [6] rely on complex feature engineering and large
amounts of labelled data, with model performance sus-
ceptible to data quality and distribution shifts. Statistical
methods [7] have high computational complexity and are
particularly sensitive to data sparsity. Although new LLM-
driven methods such as SAC-KG [8] and KGGen [9] are
emerging, their effectiveness remains to be validated, and
there is a lack of unified evaluation metrics.
This study is dedicated to the construction of document-
level knowledge graphs, with the assumption that each doc-
ument corresponds to an ideal knowledge graph. Based on
this assumption, we have established a quantitative evaluation
system. Specifically, in operationalizing the concept of “clos-
est,” we employ a dual evaluation criterion: First, concerning
the topological structure, the constructed knowledge graph
must comprehensively encompass all nodes present in the
ideal knowledge graph. Second, regarding the relationship
networks, for each corresponding node, its associated struc-
ture must attain maximum similarity with the topological
relationships of the corresponding node in the ideal knowl-
edge graph. This dual-constraint mechanism ensures both
the completeness of knowledge elements and the fidelity
of semantic relationships, thereby providing a quantifiable
theoretical framework for assessing the quality of knowl-
edge graphs. To achieve the aforementioned objectives, we
propose the RAKG framework, which effectively addresses
the two core issues of topological structure coverage and
relationship network alignment.
Topological Structure Coverage: We employ a sentence-
by-sentence Named Entity Recognition (NER) approach,
fully capitalizing on the robust natural language processing
capabilities of LLMs. In the course of this sentence-by-
sentence analysis, LLMs are capable of nearly perfectly
identifying entities within the text, thereby ensuring the com-
pleteness of nodes in the knowledge graph. These identified
entities, acting as pre-entities, provide a solid foundation for
the subsequent construction of the relationship network.
Relationship Network Alignment: For the construction
of relationship networks, the relationship network of each
node in the ideal knowledge graph is derived from the
integration of all text segments where the node appears.
Therefore, we propose the following two-step strategy: (1)
Corpus Retrospective Retrieval. By retroactively retrieving
the text segments where identified entities appear, we in-
tegrate multi-perspective semantic information and input it
into the LLM for relationship network generation. (2) GrapharXiv:2504.09823v1  [cs.IR]  14 Apr 2025

Structure Retrieval. To maintain consistency with the initial
knowledge graph, we further retrieve relevant information
about the node from the initial graph and integrate this graph
information into the input of the LLM.
Through these methods, the RAKG framework achieves
high similarity between the relationship networks of each
node in the knowledge graph and those in the ideal knowl-
edge graph while maintaining consistency with existing
knowledge. The main contributions of RAKG are as follows:
•RAKG provides a comprehensive end-to-end solution
for constructing knowledge graphs from documents. It
encompasses the entire process and enables a greater
focus on contextual information than traditional frag-
mented frameworks.
•RAKG introduces a progressive knowledge extraction
method that is predicated on the concept of pre-entities.
These pre-entities serve as intermediate representation
units, and information integration is performed based
on them. This approach effectively mitigates the com-
plexity associated with entity disambiguation and cir-
cumvents the long-distance forgetting issue inherent in
LLMs.
•In evaluating knowledge graph quality, RAKG is the
first to introduce the RAG evaluation framework into
the domain of knowledge graph construction. Addi-
tionally, it develops standard knowledge graphs and
corresponding evaluation methods, thereby facilitating
the practical assessment of the quality of constructed
knowledge graphs.
•The proposed method shows promising performance on
the MINE dataset [9]; meanwhile, the related codes have
been open-sourced to benefit the community.
II. R ELATED WORKS
Traditional KGC methods rely on expert systems and
rule-based pattern matching [10]. While these methods can
ensure a certain level of knowledge accuracy, they face high
labour costs and poor scalability. With the development of
deep learning, end-to-end construction methods based on
neural networks have significantly improved the efficiency of
relation extraction. In particular, the rapid growth of LLMs
has provided the technical foundation for automated and
large-scale knowledge graph construction [11], [12].
In the field of KGC, existing research mainly falls into
two categories. One focuses on extracting triples from sen-
tences [10], addressing named entity recognition (NER) and
relation establishment within sentences, yet ignoring entity
associations across sentences. The other targets are building
knowledge graphs from document-level text [13], tackling
NER, relation extraction, and entity resolution. With the
rapid development of LLMs, more studies are using them
for document-level knowledge graph construction. However,
due to the context limitations of LLMs, simple applications
often fail to meet expectations.A. Named Entity Recognition
NER (Named Entity Recognition) is the foundational step
in KGC, aiming to identify entities with specific meanings
from text, such as names of people, places, and organizations.
Traditional methods mainly rely on rule-based matching and
dictionary lookups, but rule formulation is complex and
challenging to adapt to different domains. With the devel-
opment of machine learning, supervised learning methods
such as Hidden Markov Models (HMMs) and Conditional
Random Fields (CRFs) have been widely applied [14],
learning entity features from annotated data. However, these
methods are highly dependent on annotated data. In recent
years, deep learning methods such as Recurrent Neural
Networks (RNNs), Convolutional Neural Networks (CNNs)
[15], and their variants have shown excellent performance
in entity recognition, automatically learning text features
to improve recognition accuracy and robustness. Moreover,
applying pre-trained language models such as BERT and
XLNet has further advanced entity recognition [16], with
their rich language knowledge and contextual understanding
capabilities supporting more accurate entity identification.
B. Coreference Resolution
Coreference resolution is a key task in knowledge graph
construction, aiming to identify different expressions in the
text that refer to the same entity and to resolve the ambiguity
of entity references [17]. Rule-based methods perform coref-
erence resolution by defining lexical and syntactic rules, but
these rules are complicated and exhausting and are easily
affected by text variations. Statistical methods train models
using corpus statistical features, but feature engineering
is complex, and generalization ability is limited. Machine
learning methods such as Support Vector Machines (SVMs)
and decision trees judge coreference by extracting feature
vectors, but feature selection and parameter optimization
are challenging. Deep learning methods [18], [19], such
as neural network models, automatically learn text features
and enhance resolution effects by incorporating attention
mechanisms. The application of pre-trained models further
strengthens their semantic understanding and coreference
relation capture capabilities [20], providing new pathways
for more accurate coreference resolution.
C. Relation Extraction
Relation extraction is a core task in knowledge graph
construction, aiming to identify relationships between en-
tities in text. Rule-based methods define pattern-matching
relationships, but rule construction is cumbersome and
challenging to adapt to different domains and text styles.
Deep learning methods [21], such as Convolutional Neural
Networks (CNNs) [22], [23], Recurrent Neural Networks
(RNNs), and their improved models, are widely used and can
automatically learn text features to improve the accuracy and
efficiency of relation extraction. Moreover, the application
of pre-trained language models such as BERT [16] and
RoBERTa has further enhanced the performance of relation
extraction, with their rich language knowledge and semantic

TABLE I
SUMMARY OF NOTATION
Notation Description
KG∗, V∗, E∗Ideal knowledge graph, its set of entities, and
its set of directed edges
KG′, V′, E′Initial knowledge graph, its set of entities, and
its set of directed edges
KG, V, E Constructed knowledge graph, its set of enti-
ties, and its set of directed edges
D Input document
T Set of text chunks derived from the document
texti An individual text chunk
e An entity
Preentity Set of preliminary entities identified by NER
from the text chunks T
VT Set of vectors representing the text chunks in
T
Vkg Set of vectors representing the entities in the
knowledge graph
VPreentity Set of vectors representing the preliminary
entities identified by NER
V ect (·) Function that vectorizes the input
understanding capabilities providing support for more precise
capture of relationships between entities [24]. Meanwhile,
researchers continuously explore new model architectures
and training strategies to improve relation extraction’s per-
formance and generalization ability.
D. Retrieval-Augmented Generation
Due to the lengthy document context in constructing
document-level knowledge graphs, directly using LLMs
makes it challenging to capture detailed information [25].
To address this issue, we introduce the RAG (Retrieval-
Augmented Generation). RAG retrieves relevant information
from external knowledge bases. It inputs it as a prompt to
LLMs, enhancing the models’ ability to handle knowledge-
intensive tasks, such as question answering, text summa-
rization, and content generation. RAG technology typically
utilizes documents and knowledge graphs as external knowl-
edge bases (e.g., lightRAG [26], GraphRAG). Innovatively,
we apply the RAG concept in reverse to the knowledge graph
construction process. By retrieving relevant passages and
graph information, we help LLMs more accurately generate
entities and their relational networks.
III. M ETHOD
We have developed a document-level knowledge graph
construction framework, RAKG, which leverages LLMs for
document-level knowledge graph construction. To facilitate
comprehension, Table I provides a summary of the key
notations used in this paper. An overview of RAKG is shown
in Figure 1.A. Problem Formulation
Given a document D, we assume the existence of a
theoretically perfect knowledge graph construction process:
KG∗= Construct( D) (1)
This ideal knowledge graph can be formally represented as:
G∗= (V∗, E∗) (2)
KG∗={(h∗
i, r∗
i, t∗
i)|h∗
i, t∗
i∈V∗, r∗
i∈E∗} (3)
Here, the set of triples KG∗comprehensively covers all
semantic relationships in document D.
The objective of this paper is to construct a knowledge
graph:
KG= RAKG( D) (4)
Its formal definition is:
G= (V, E) (5)
KG={(hi, ri, ti)|hi, ti∈V, r i∈E} (6)
The constructed knowledge graph KG =RAKG (D)
must satisfy the following approximation conditions:
∀e∗∈V∗∃e∈V (7)
e∗≈e (8)
rel(e∗)≈rel(e) (9)
The relationship mapping functions are defined as:
rel(e∗) ={(e∗, r∗
i, t∗
i)|(e∗, r∗
i, t∗
i)∈KG∗} (10)
rel(e) ={(e, ri, ti)|(e, ri, ti)∈KG} (11)
B. Knowledge Base Vectorization
1) Document Chunking and Vectorization: RAKG em-
ploys a dynamic chunking strategy based on semantic in-
tegrity rather than fixed-length divisions. Specifically, the text
is segmented at sentence boundaries as shown in Equation
(12,13), and each chunk is vectorized as shown in Equation
(14).
T=DocSplit (D) (12)
s.t.∀i̸=j, text i∩textj=∅ ∧n[
i=1texti=D (13)
VT={⃗ vi|⃗ vi=V ect (texti), text i∈T} (14)
This approach reduces the amount of information processed
by the LLM each time while ensuring the semantic integrity
of each chunk, thereby improving the accuracy of named
entity recognition.
2) Knowledge Graph Vectorization: The initial knowl-
edge graph is vectorized by extracting the name and type of
each node and using the BGE-M3 model [27] for vectoriza-
tion.
Vkg={⃗ vj|⃗ vj=V ect (node j), node j∈KG′} (15)

Fig. 1. The RAKG framework processes documents through sentence segmentation and vectorization, extracts preliminary entities, and performs entity
disambiguation and vectorization. The processed entities undergo Corpus Retrospective Retrieval to obtain relevant texts and Graph Structure Retrieval to
get related KG. Subsequently, LLM is employed to integrate the retrieved information for constructing relation networks, which are merged for each entity.
Finally, the newly built knowledge graph is combined with the original one.
C. Pre-Entity Construction
1) Entity Recognition and Vectorization: Named entity
recognition is performed segment by segment for the seg-
mented text chunks. This process is completed by the LLM,
which analyzes the entire text chunk to identify entities. For
each pre-entity, the LLM further assigns type and description
attributes. The type attribute distinguishes the category of the
entity, while the description provides a brief explanation to
differentiate entities with similar names.
Preentity i=NER(texti) (16)
Preentity =[
iPreentity i (17)
(18)
A chunk-id attribute is added to indicate which text chunk
the entity originates from. The entity’s name and type are
combined and vectorized.
VPreentity ={⃗ vi=V ect (ei)|ei∈Preentity} (19)
2) Entity Disambiguation: After completing entity recog-
nition and vectorization for the entire document, similarity
checks are performed on each entity. Entities with similarity
scores above a threshold are placed into a preliminary similar
entity set, which is then individually inspected by the LLM to
obtain the final similar entity set. Entities in the final set aredisambiguated into a single entity, with their corresponding
chunk-ids linked together.
Sim(ei) ={ej|ej∈Preentity , V ectJudge (ei, ej) = 1}
(20)
Same (ei) ={ej|ej∈Sim(ei), SameJudge (ei, ej) = 1}
(21)
These functions are used sequentially in the disambiguation
process. First, VectJudge (ei, ej)is applied to efficiently filter
potential matches based on vector similarity, forming a
preliminary similar entity set as shown in Equation (20).
Then, SameJudge (ei, ej)is used to refine this set by making
a final determination of identity, resulting in the final similar
entity set as shown in Equation (21).
D. Relationship Network Construction
1) Corpus Retrospective Retrieval: For a specified entity,
retrieve the associated text segments via chunk-id and use
vector retrieval to obtain text segments similar to the selected
entity.
retriever VT(e) =(
textitextj∈T,
retriever (e,vj, threshold ) = 1)
(22)
2) Graph Structure Retrieval: Perform vector retrieval
in the initial knowledge graph for a specified entity to

obtain entities similar to the selected entity and extract their
relationship networks.
retriever Vkg(e) =(
node jnode j∈KG′,
retriever (e,vj, threshold ) = 1)
(23)
3) Relationship Network Generation and Evaluation:
Integrate the retrieved text and relationship networks and
process this information using the LLM to obtain attributes
and relationships for the central entity as shown in Equation
(24). Use the LLM as a judge for the generated triplets to
assess their truthfulness.
rel(ei) =LLM rel(entity 1, retriever text(ei), retriever kg(ei))
(24)
E. Knowledge Graph Fusion
1) Entity Merging: Entities in the new knowledge graph
may refer to the same entities in the initial knowledge graph.
It is necessary to disambiguate and merge entities from the
new knowledge graph with those in the initial knowledge
graph.
2) Relationship Integration: To obtain a more compre-
hensive knowledge graph, relationships in the new knowl-
edge graph need to be integrated with those in the initial
knowledge graph.
IV. E XPERIENCE
To comprehensively evaluate the performance of RAKG
across different topics and domains, we conducted experi-
ments on the MINE dataset [9].
A. Experimental Settings
1) Dataset: The MINE dataset contains 105 articles, each
approximately 1000 words in length, covering multiple do-
mains, including history, art, science, ethics, and psychology.
An LLM generated these articles based on 105 diverse
topics. To assess the quality of KG generation, we used
the semantic inclusion rate to measure its ability to capture
key information from the articles. Specifically, we extracted
15 facts for each article using the LLM and manually
verified their accuracy and relevance. By checking whether
the knowledge graph captured these facts, we evaluated the
effectiveness of the text-to-KG extractor.
We queried the article’s knowledge graph for the 15 facts
for each KG generation method. We identified the top k
nodes semantically closest to each point, as well as all
nodes within two hops of these nodes. These nodes and their
relationships were returned as query results. We then used
the LLM to evaluate the results and generate binary outputs:
if the fact could be inferred based solely on the queried nodes
and relationships, the output was 1; otherwise, it was 0. The
final MINE score for each knowledge graph generator was
calculated by determining the proportion of 1s among all 15
evaluations, objectively comparing each method’s ability to
capture and retrieve information from the articles accurately.2) Baseline:
•KGGen [9] : The Stanford Trustworthy AI Research
Laboratory (STAIR Lab) developed KGGen, an open-
source tool designed to generate knowledge graphs from
plain text automatically. It leverages advanced language
models and clustering algorithms to transform unstruc-
tured text data into structured networks of entities and
relationships. Available as a Python library (installable
viapip install kg-gen ), KGGen is convenient
for researchers and developers to use.
•GraphRAG [3] : Graph-based Retrieval-Augmented
Generation (GraphRAG) is a knowledge graph-based
enhanced retrieval generation framework proposed by
Microsoft, aiming to overcome the limitations of tradi-
tional RAG methods. Its core idea is to build a struc-
tured knowledge graph to model the global semantics of
document content, thereby enhancing the performance
of LLMs. The workflow of GraphRAG mainly includes
three steps: graph indexing (building and indexing the
knowledge graph), graph retrieval (retrieving relevant
information from the knowledge graph), and graph-
enhanced generation (generating text using the retrieved
information). This method effectively utilizes the struc-
tural information between entities, enhancing the rele-
vance and accuracy of the generated results.
3) Metrics:
•Entity Density (ED) :
ED=Ne (25)
Nerepresents the number of entities. Generally, the
more entities extracted, the stronger the ability to extract
information from the text.
•Relationship Richness (RR) :
RR=Nr
Ne(26)
Nrrepresents the number of relationships (attributes).
Generally, the more complex the entity relationship
network, the stronger the ability to extract information
from the text.
•Entity Fidelity (EF) :
EF=1
NeNeX
i=1LLMJudge entity (ei, retriever VT(ei))
(27)
LLM is used as a judge to evaluate each extracted entity
and assign a value between 0 and 1, representing the
entity’s credibility.
•Relationship Fidelity (RF) :
RF=1
NrNrX
i=1LLMJudge rel
(ei, retriever VT(ei), retriever Vkg(ei))(28)
LLM is used as a judge to evaluate each extracted
relationship and assign a value between 0 and 1, repre-
senting the credibility of the relationship.

•accuracy : The accuracy of question answering on the
MINE dataset is measured by the constructed knowl-
edge graph, where higher accuracy means the graph
retains the semantic info better than the original text.
•Entity Coverage (EC) :
EC=|E∩E∗|
|E∗|(29)
The entity coverage rate reflects the proportion of enti-
ties in the evaluated knowledge graph that semantically
match those in the standard knowledge graph, indicating
its completeness at the entity level. It is calculated by
dividing the number of entities in the intersection of the
evaluated and standard knowledge graph entity sets by
the number of entities in the standard knowledge graph
entity set.
•Relation Network Similarity (RNS) :
RNS =X
ei∈E∩E∗(RelSim i×EntityWeight i)(30)
Relationship network similarity measures the similarity
between the evaluated and standard knowledge graphs at
the relationship level by calculating the similarity of the
relationship networks corresponding to the same entities
and combining it with entity weights. For each entity in
the intersection of the evaluated and standard knowledge
graphs, the relationship similarity is calculated and
multiplied by the corresponding entity weight. Then,
the results for all entities are summed.
B. Experimental Results
We conducted experiments on the MINE dataset, where
the LLM we used was Qwen2.5-72B [28], and the vector
embedding model was BGE-M3 [27].
Fig. 2. Distribution of SC scores across 105 articles for GraphRAG,
KGGen, and RAKG on the MINE dataset. The results demonstrate that
RAKG achieves an accuracy of 95.81%, outperforming KGGen (86.48%)
and GraphRAG (89.71%).
Compared with baseline models such as KGGen, the
RAKG model has demonstrated a significant improvement
in the key metric of accuracy as Figure 2, with a clear
and distinct advantage. This result strongly indicates that the
knowledge graph constructed by the RAKG model possesses
a more substantial capability for semantic information ex-
traction, enabling it to more comprehensively and accuratelymine rich semantic knowledge from the original text. With
this excellent ability in knowledge extraction and represen-
tation, the RAKG model has the potential to achieve better
performance in a variety of subsequent natural language
processing tasks, such as semantic search and intelligent
question answering.
Fig. 3. This visualisation of the experimental results shows the entity
density and relation richness of knowledge graphs generated by RAKG,
GraphRAG, and KGGen. The results indicate that RAKG produces more
dense entities and richer relations than GraphRAG and KGGen.
The knowledge graph constructed by the RAKG model
demonstrates a higher concentration in entity density as
Figure 3. This means that, against the same textual corpus
background, RAKG can identify and incorporate a piece of
richer entity information, fully excavating those entities with
key semantic values hidden deep within the text. As a result,
the entity composition of the knowledge graph becomes more
substantial and dense. At the same time, on the dimension of
relationship density as Figure 3, the knowledge graph built
by RAKG exhibits a more complex network of associations.
It can accurately capture the diverse and nuanced interaction
relationships between entities and organize and present the
intricate relational threads implicit in the textual context, thus
weaving a dense and complex web of relationships. These
significant characteristics indicate that, in the knowledge
graph construction process, the RAKG model can more
efficiently mine key information from the text and more
comprehensively capture entities and their relationships than
other models.
Fig. 4. The process of LLM as judge: Extracted entities are checked against
the source text to eliminate hallucinations. The retriever uses entities to
fetch relevant texts and KG, building a relation network. This network is
then verified for consistency with the retrieved information.
In constructing and evaluating the knowledge graph, we

innovatively introduced the evaluation metric system of RAG
into the KGC task to ensure the accuracy and fidelity of the
constructed knowledge graph. Specifically, we leveraged the
powerful capabilities of LLMs to rigorously assess the enti-
ties extracted by RAKG from the text, determining whether
these entities strictly adhere to the original text’s content
framework and semantic logic. Meanwhile, the LLM was
also employed to evaluate whether RAKG maintained high
fidelity to the text when constructing the relationship network
based on retriever content, ensuring that the established
relationships reflect the semantic associations within the text
rather than being fabricated out of thin air. Through this dual
evaluation mechanism, we could precisely eliminate nodes
and relationships that were falsely generated due to potential
hallucinations of the LLM, thereby effectively enhancing the
quality and reliability of the knowledge graph. The detailed
implementation process of this approach is illustrated in
Figure 4.
Fig. 5. Results of LLM as judge: The pass rate for entities is around
91.33%, and the pass rate for relation networks is approximately 94.51%.
In evaluating the RAKG model, we employed the Deep-
Eval platform [29] as our assessment tool. It is evident
from the results that the output distrust rate of the RAKG
model consistently remains at an extremely low level. This
stable performance powerfully demonstrates the reliability
and consistency of the RAKG model in generating knowl-
edge graphs. Even in the rare instances where hallucinations
from the LLM lead to the creation of a small number of
false nodes or relationships, we can effectively identify and
eliminate these potential errors through the ”LLM as judge”
mechanism. This process is clearly and intuitively illustrated
in Figure 5.
We conducted an in-depth model performance analysis
from two key dimensions: Entity Coverage (EC) and Relation
Similarity (RNS). According to the data in Table II, the
RAKG model achieved an entity coverage of 0.8752 (stan-
dard deviation: 0.1047) and a relation similarity of 0.7998
(standard deviation: 0.0912). Both metrics significantly out-
performed other models. GraphRAG achieved an entity cov-
erage of 0.6438 and a relation similarity of 0.7278, whileTABLE II
MERGED MATRIX WITH STANDARD KG R EFERENCE
Targetideal KG as Reference
EC RNS
KGGen 0.6020 ±0.1754 0.6321 ±0.0818
GraphRAG 0.6438 ±0.1558 0.7278 ±0.0752
RAKG 0.8752 ±0.1047 0.8752 ±0.1047 0.8752 ±0.1047 0.7998 ±0.0912 0.7998 ±0.0912 0.7998 ±0.0912
KGGen achieved an entity coverage of 0.6020 and a relation
similarity of 0.6321. RAKG demonstrated high consistency
with the standard knowledge graph across both dimensions,
significantly outperforming GraphRAG and KGGen. Further
analysis revealed that RAKG achieved the highest entity
coverage and exhibited a minor standard deviation, indicating
more excellent stability in its results. In terms of relation
similarity, RAKG maintained a significant leading advantage,
demonstrating its ability to capture relationships between en-
tities more accurately during knowledge graph construction.
These results indicate that RAKG has substantial benefits in
both the completeness and accuracy of knowledge graph con-
struction, making it better suited for high-quality knowledge
graph requirements.
V. C ASE STUDY
In the case study, we used an article titled ”The Life
Cycle of a Butterfly” as the application scenario to compare
the performance of the RAKG framework with the baseline
models, as shown in Figure 6. RAKG’s named entity recog-
nition module detected 23 core entities in the article, with
”Butterfly Egg,” ”Caterpillar,” and ”Adult Butterfly” being
central. These entities have dense text blocks in the article,
indicating key concepts. Focusing on ”Adult Butterfly,” we
retrieved professional text chunks describing five features.
We also obtained a sub-graph related to ”Adult Butterfly”
from the original KG via graph-structured retrieval. After
NER, Corpus Retrospective Retrieval, and Graph Structure
Retrieval, we integrated each entity’s text blocks with their
sub-graphs. These integrated data were fed into a LLM to
construct relationship networks. The LLM analyzed the text
blocks and sub-graphs to generate entity-specific relationship
networks, forming complete sub-graphs. By integrating all
sub-graphs, we built a systematic, structured knowledge
graph that clearly shows the article’s core concepts and their
relationships.
To further evaluate the performance of the RAKG frame-
work, we compared the knowledge graph it generated with
the results of baseline models. The results showed that the
knowledge graph constructed by RAKG has higher EC and
RMS, making it the most similar to the ideal knowledge
graph. For example, focusing on the relationship network
of the entity ”Adult Butterfly”, RAKG retrieved the pas-
sage ”Adult butterflies feed on nectar from flowers using
their long, tubular mouthparts called proboscis. They play
a crucial role in pollination by transferring pollen from one
flower to another, helping plants reproduce.” From this, it
was concluded that adult butterflies contribute to pollination,

chunk chunk
chunk chunk
Pre-existing Knowledge 
Graph
centered on Larvae
Integration of information 
centered on butterfliescentered on eggs
Butterfly -centered knowledge graphLarvae -centered knowledge graphEggs -centered knowledge graph
KG1 KG2
Extracting…
Text1
Wing Membrane 
Hardening 
MechanismText2
Energy Acquisition 
Organ AdaptabilityCorpus 
Retrospective 
Retrieval
Graph 
Structure 
Retrieval.Fig. 6. Case Study. Different colors in the document represent different entities involved. The document is segmented and generates pre-entities such as
Butterfly, Caterpillar, and Butterfly Egg. Taking the Butterfly as an example, the texts are retrieved from the text corpus, and related nodes are retrieved
from the pre-existing knowledge graph. After integrating the information, the LLM extracts a knowledge graph centered on the butterfly. Finally, these
knowledge graphs are integrated.
resulting in the triple ”Adult butterfly”-”contributes to”-
”POLLINATION.” This demonstrates that RAKG can more
comprehensively capture the complex relationships between
various entities involved in the different life stages of but-
terflies.
VI. CONCLUSION
In this paper, we propose a novel document-level knowl-
edge graph construction framework named RAKG, which
can directly transform document corpora into knowledge
graphs. RAKG employs a progressive knowledge extraction
method based on pre-entities to integrate information. This
approach effectively reduces the complexity of entity dis-
ambiguation, circumvents the long-distance forgetting issues
of LLMs, and achieves near-perfect performance regard-
ing topological structure coverage and relationship network
alignment. The superior performance of RAKG compared to
existing state-of-the-art methods demonstrates the effective-
ness of our proposed framework.
REFERENCES
[1] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin,
C. Zhou, and J. Zhou, “Qwen-vl: A versatile vision-language model
for understanding, localization, text reading, and beyond,” 2023.
[Online]. Available: https://arxiv.org/abs/2308.12966
[2] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel, and
D. Kiela, “Retrieval-augmented generation for knowledge-intensive
nlp tasks,” in Advances in Neural Information Processing Systems ,
H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin,
Eds., vol. 33. Curran Associates, Inc., 2020, pp. 9459–9474.
[Online]. Available: https://proceedings.neurips.cc/paper files/paper/
2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
[3] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody,
S. Truitt, D. Metropolitansky, R. O. Ness, and J. Larson, “From local
to global: A graph rag approach to query-focused summarization,”
2025. [Online]. Available: https://arxiv.org/abs/2404.16130
[4] J. Wang, J. Fu, R. Wang, L. Song, and J. Bian, “Pike-rag: specialized
knowledge and rationale augmented generation,” 2025. [Online].
Available: https://arxiv.org/abs/2501.11551[5] T. R. Gruber, “A translation approach to portable ontology specifica-
tions,” Knowledge acquisition , vol. 5, no. 2, pp. 199–220, 1993.
[6] G. Lample, M. Ballesteros, S. Subramanian, K. Kawakami, and
C. Dyer, “Neural architectures for named entity recognition,” 2016.
[Online]. Available: https://arxiv.org/abs/1603.01360
[7] O. Etzioni, M. Banko, S. Soderland, and D. S. Weld, “Open informa-
tion extraction from the web,” Communications of the ACM , vol. 51,
no. 12, pp. 68–74, 2008.
[8] H. Chen, X. Shen, Q. Lv, J. Wang, X. Ni, and J. Ye,
“Sac-kg: Exploiting large language models as skilled automatic
constructors for domain knowledge graphs,” 2024. [Online]. Available:
https://arxiv.org/abs/2410.02811
[9] B. Mo, K. Yu, J. Kazdan, P. Mpala, L. Yu, C. Cundy,
C. Kanatsoulis, and S. Koyejo, “Kggen: Extracting knowledge graphs
from plain text with language models,” 2025. [Online]. Available:
https://arxiv.org/abs/2502.09956
[10] F. M. Suchanek, G. Kasneci, and G. Weikum, “Yago: a core of seman-
tic knowledge,” in Proceedings of the 16th international conference
on World Wide Web , 2007, pp. 697–706.
[11] B. Zhang and H. Soh, “Extract, define, canonicalize: An llm-
based framework for knowledge graph construction,” arXiv preprint
arXiv:2404.03868 , 2024.
[12] Y . Lu and J. Wang, “Karma: Leveraging multi-agent llms for
automated knowledge graph enrichment,” 2025. [Online]. Available:
https://arxiv.org/abs/2502.06472
[13] L. Zhong, J. Wu, Q. Li, H. Peng, and X. Wu, “A comprehensive
survey on automatic knowledge graph construction,” ACM Computing
Surveys , vol. 56, no. 4, pp. 1–62, 2023.
[14] R. Panchendrarajan and A. Amaresan, “Bidirectional lstm-crf for
named entity recognition.” 32nd Pacific Asia Conference on Lan-
guage, Information and Computation, 2018.
[15] N. Zhang, X. Xu, L. Tao, H. Yu, H. Ye, S. Qiao, X. Xie, X. Chen,
Z. Li, L. Li, X. Liang, Y . Yao, S. Deng, P. Wang, W. Zhang,
Z. Zhang, C. Tan, Q. Chen, F. Xiong, F. Huang, G. Zheng, and
H. Chen, “Deepke: A deep learning based knowledge extraction
toolkit for knowledge base population,” 2023. [Online]. Available:
https://arxiv.org/abs/2201.03335
[16] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training
of deep bidirectional transformers for language understanding,” in
Proceedings of the 2019 conference of the North American chapter
of the association for computational linguistics: human language
technologies, volume 1 (long and short papers) , 2019, pp. 4171–4186.
[17] L. Miculicich and J. Henderson, “Graph refinement for coreference
resolution,” arXiv preprint arXiv:2203.16574 , 2022.
[18] K. Lee, L. He, M. Lewis, and L. Zettlemoyer, “End-to-end neural
coreference resolution,” arXiv preprint arXiv:1707.07045 , 2017.

[19] J.-L. Wu and W.-Y . Ma, “A deep learning framework for coreference
resolution based on convolutional neural network,” in 2017 IEEE 11th
International Conference on Semantic Computing (ICSC) . IEEE,
2017, pp. 61–64.
[20] M. Joshi, O. Levy, D. S. Weld, and L. Zettlemoyer, “Bert
for coreference resolution: Baselines and analysis,” arXiv preprint
arXiv:1908.09091 , 2019.
[21] D. Zeng, K. Liu, S. Lai, G. Zhou, and J. Zhao, “Relation classification
via convolutional deep neural network,” in Proceedings of COLING
2014, the 25th international conference on computational linguistics:
technical papers , 2014, pp. 2335–2344.
[22] T. H. Nguyen and R. Grishman, “Relation extraction: Perspective from
convolutional neural networks,” in Proceedings of the 1st workshop on
vector space modeling for natural language processing , 2015, pp. 39–
48.
[23] C. N. d. Santos, B. Xiang, and B. Zhou, “Classifying relations
by ranking with convolutional neural networks,” arXiv preprint
arXiv:1504.06580 , 2015.
[24] Y . Cui, W. Che, T. Liu, B. Qin, S. Wang, and G. Hu, “Revisiting
pre-trained models for chinese natural language processing,” arXiv
preprint arXiv:2004.13922 , 2020.
[25] R. Yang, B. Yang, A. Feng, S. Ouyang, M. Blum, T. She, Y . Jiang,
F. Lecue, J. Lu, and I. Li, “Graphusion: A rag framework for
knowledge graph construction with a global perspective,” 2025.
[Online]. Available: https://arxiv.org/abs/2410.17600
[26] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “Lightrag: Simple and
fast retrieval-augmented generation,” 2024.
[27] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, “Bge
m3-embedding: Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation,” 2024. [Online].
Available: https://arxiv.org/abs/2402.03216
[28] Qwen, :, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu,
C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang,
J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang,
L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li,
T. Tang, T. Xia, X. Ren, X. Ren, Y . Fan, Y . Su, Y . Zhang, Y . Wan,
Y . Liu, Z. Cui, Z. Zhang, and Z. Qiu, “Qwen2.5 technical report,”
2025. [Online]. Available: https://arxiv.org/abs/2412.15115
[29] J. Ip and K. V ongthongsri, “deepeval,” Mar. 2025. [Online]. Available:
https://github.com/confident-ai/deepeval