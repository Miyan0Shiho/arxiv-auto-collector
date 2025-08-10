# mKG-RAG: Multimodal Knowledge Graph-Enhanced RAG for Visual Question Answering

**Authors**: Xu Yuan, Liangbo Ning, Wenqi Fan, Qing Li

**Published**: 2025-08-07 12:22:50

**PDF URL**: [http://arxiv.org/pdf/2508.05318v1](http://arxiv.org/pdf/2508.05318v1)

## Abstract
Recently, Retrieval-Augmented Generation (RAG) has been proposed to expand
internal knowledge of Multimodal Large Language Models (MLLMs) by incorporating
external knowledge databases into the generation process, which is widely used
for knowledge-based Visual Question Answering (VQA) tasks. Despite impressive
advancements, vanilla RAG-based VQA methods that rely on unstructured documents
and overlook the structural relationships among knowledge elements frequently
introduce irrelevant or misleading content, reducing answer accuracy and
reliability. To overcome these challenges, a promising solution is to integrate
multimodal knowledge graphs (KGs) into RAG-based VQA frameworks to enhance the
generation by introducing structured multimodal knowledge. Therefore, in this
paper, we propose a novel multimodal knowledge-augmented generation framework
(mKG-RAG) based on multimodal KGs for knowledge-intensive VQA tasks.
Specifically, our approach leverages MLLM-powered keyword extraction and
vision-text matching to distill semantically consistent and modality-aligned
entities/relationships from multimodal documents, constructing high-quality
multimodal KGs as structured knowledge representations. In addition, a
dual-stage retrieval strategy equipped with a question-aware multimodal
retriever is introduced to improve retrieval efficiency while refining
precision. Comprehensive experiments demonstrate that our approach
significantly outperforms existing methods, setting a new state-of-the-art for
knowledge-based VQA.

## Full Text


<!-- PDF content starts -->

mKG-RAG: Multimodal Knowledge Graph-Enhanced
RAG for Visual Question Answering
Xu Yuan
Department of Computing
The Hong Kong Polytechnic University
Hong Kong SAR, China
xuyuan127@gmail.comLiangbo Ning
Department of Computing
The Hong Kong Polytechnic University
Hong Kong SAR, China
BigLemon1123@gmail.com
Wenqi Fan∗
Department of Computing
The Hong Kong Polytechnic University
Hong Kong SAR, China
wenqifan03@gmail.comQing Li*
Department of Computing
The Hong Kong Polytechnic University
Hong Kong SAR, China
qing-prof.li@polyu.edu.hk
Abstract
Recently, Retrieval-Augmented Generation (RAG) has been proposed to expand
internal knowledge of Multimodal Large Language Models (MLLMs) by incorpo-
rating external knowledge databases into the generation process, which is widely
used for knowledge-based Visual Question Answering (VQA) tasks. Despite im-
pressive advancements, vanilla RAG-based VQA methods that rely on unstructured
documents and overlook the structural relationships among knowledge elements
frequently introduce irrelevant or misleading content, reducing answer accuracy
and reliability. To overcome these challenges, a promising solution is to integrate
multimodal knowledge graphs (KGs) into RAG-based VQA frameworks to en-
hance the generation by introducing structured multimodal knowledge. Therefore,
in this paper, we propose a novel multimodal knowledge-augmented generation
framework ( mKG-RAG ) based on multimodal KGs for knowledge-intensive VQA
tasks. Specifically, our approach leverages MLLM-powered keyword extraction
and vision-text matching to distill semantically consistent and modality-aligned
entities/relationships from multimodal documents, constructing high-quality mul-
timodal KGs as structured knowledge representations. In addition, a dual-stage
retrieval strategy equipped with a question-aware multimodal retriever is introduced
to improve retrieval efficiency while refining precision. Comprehensive experi-
ments demonstrate that our approach significantly outperforms existing methods,
setting a new state-of-the-art for knowledge-based VQA.
1 Introduction
Visual Question Answering (VQA) [ 2,19] is a challenging task at the intersection of vision and
language understanding, requiring models to interpret images and answer related questions. This ca-
pability has enabled remarkable advances in various domains, including medical image diagnosis [ 33]
and customer service support [ 12]. Recently, due to the powerful visual-linguistic understanding and
reasoning capabilities, Multimodal Large Language Models ( MLLMs ) [35,30,50,7] have provided
a promising solution to conventional VQA tasks. For instance, LLaV A [ 35] demonstrates strong
∗Corresponding author
Preprint. Under review.arXiv:2508.05318v1  [cs.CV]  7 Aug 2025

zero-shot performance on commonsense VQA by leveraging pre-trained visual encoders for image
representation alongside the reasoning capabilities of large language models (LLMs). Despite no-
table advancements, MLLMs face critical limitations in knowledge-intensive VQA scenarios [ 40,6]
(termed knowledge-based VQA ), particularly those requiring encyclopedic knowledge, long-tail
factual recall, or contextual reasoning beyond immediate visual inputs. As illustrated in Figure 1 (a),
when queried about the latest renovation date of a stadium, typical MLLMs exhibit two characteristic
failure modes: generating plausible but factually incorrect responses or refusing to answer altogether.
These issues stem from the scarcity of relevant knowledge in MLLMs’ training corpus and the
inherent difficulty of memorizing low-frequency facts [6].
Recently, Retrieval-Augmented Generation ( RAG ) [16] has shown great potential in addressing
these challenges by leveraging external knowledge databases to supplement the internal knowledge
of MLLMs, thereby enabling more accurate answer generation [ 32,4,11]. Specifically, multiple
query-relevant documents are retrieved from external knowledge databases and serve as in-context
information to augment the generation process of MLLMs. Despite their success, vanilla RAG-
based VQA methods that rely on unstructured documents or paragraphs often introduce irrelevant
or even misleading information [ 38,51], thereby compromising the accuracy and reliability of
generated answers. Moreover, these approaches typically overlook the structural relationships among
knowledge elements, limiting the reasoning capabilities of MLLMs. As illustrated in Figure 1 (b),
the presence of noisy and unstructured context makes it difficult for MLLMs to identify and leverage
relevant supporting evidence. To overcome these limitations, a promising direction is to retrieve
structured knowledge, such as Knowledge Graphs (KGs) [ 23] for augmentation generation [ 22,
15,59]. However, in the VQA setting, which inherently involves multimodal reasoning, relying
solely on textual KGs is suboptimal, as both modalities are crucial for identifying relevant knowledge.
Therefore, integrating Multimodal Knowledge Graphs into the retrieval-augmented VQA framework
presents a more robust solution to generate reliable and precise responses in knowledge-intensive
scenarios, as illustrated in Figure 1 (c).
Visual-only
SearchI can't give 
the information.When were the latest renovations of this
stadium?  [Answer: 2010]
In 2012
Factual Err or Lack of Knowledge
(a) Knowledge-based VQA
(b) Vanilla RAG-based VQA  Framework
(c) mKG-RAG: RAG with Multimodal Knowledge Graphs In 2018
In 2010
Textual-only
Search
Multimodal
Search
Graph
Traversal
Ski 
Stadium
Marcialonga
Relation: the stadium was
renovated in 2010  for the
World Ski Championship.
Skier
Word Ski
ChampionshipCross-country Ski Stadium
The Cross country and
biathlon center Fabio Canàl,
until December 2018  named
Lago di Tésero Cross Coun-
try Stadium is a sport venue
located in the village of ...
Unstructur ed Documents
Structur ed Knowledge GraphsInaccurate Retrieval
Figure 1: Illustration of issues in knowledge-based VQA.
(b) Vanilla RA methods suffer from retrieving unstructured
knowledge from external documents via unimodal retrievers.
(c) Our mKG-RAG augments MLLMs with structural infor-
mation from multimodal knowledge graphs.However, retrieving relevant knowl-
edge from multimodal knowledge
graphs to enhance the generation of
knowledge-based VQA tasks is ex-
ceptionally challenging. First, off-
the-shelf multimodal KGs [ 36] are
generally built around common enti-
ties, and often lack the encyclopedic
or long-tail knowledge required by
knowledge-intensive questions, ren-
dering them ineffective for direct use
in knowledge-based VQA. Moreover,
current knowledge sources used in
knowledge-based VQA [ 40,6] are
typically organized in unstructured
documents containing substantial con-
textual noise, making it challenging to
extract well-structured entities and re-
lationships essential for constructing
high-quality multimodal KGs. Fur-
thermore, a large-scale knowledge
graph constructed from millions of
documents, each potentially contain-
ing hundreds of entities and relation-
ships, significantly expands the search
space. Consequently, performing di-
rect retrieval over such a graph is com-
putationally inefficient and adversely
affects retrieval precision.
To address the challenges above, this
paper proposes mKG-RAG , a novel
2

retrieval-augmented generation framework integrated with multimodal knowledge graphs designed to
enhance the reasoning capabilities of MLLMs in knowledge-based VQA tasks. More specifically, a
multimodal knowledge graph construction module is introduced to transform unstructured multimodal
documents, such as Wikipedia articles, into structured knowledge representations. This module
leverages MLLM-powered keyword extraction and vision-text alignment to extract semantically
consistent and modality-aligned entities and relationships from external multimodal documents.
To enable efficient retrieval, mKG-RAG develops a dual-stage search paradigm that combines a
coarse-grained document recall and a fine-grained entity/relationship retrieval. The coarse stage
efficiently narrows the search space by identifying candidate documents likely to contain relevant
evidence, while the fine stage refines the results by retrieving query-relevant entities and relationships
from multimodal KGs that are dynamically constructed from these potentially noisy documents.
During retrieval, unlike previous methods that rely on isolated unimodal retrievers, we introduce a
question-aware multimodal retriever trained on a high-quality question-evidence dataset to further
enhance retrieval precision within the proposed search paradigm. Comprehensive evaluations on
two frequently used benchmarks demonstrate the superior performance of mKG-RAG, achieving an
accuracy of 36.3% on E-VQA and 40.5% on InfoSeek.
The contributions of this work are summarized as follows:
•We propose mKG-RAG, a novel multimodal knowledge-augmented generation framework
that integrates RAG with multimodal KGs to enhance the knowledge reasoning of MLLMs.
To the best of our knowledge, this is the first work to investigate the potential of multimodal
knowledge graphs in knowledge-intensive VQA tasks.
•Our framework develops a multimodal KG construction pipeline, allowing the extraction
of image-text aligned entities and relations from multimodal documents. Additionally, a
dual-stage retrieval schema with a question-aware multimodal retriever enables us to unleash
the potential of RAG incorporated with multimodal KGs.
•Extensive experiments demonstrate that mKG-RAG significantly outperforms strong base-
lines, setting new state-of-the-art results on E-VQA and InfoSeek.
2 Related Work
Benefiting from the rapid advancement of Large Language Models [ 8,49,9,57,18,17,45,42],
Multimodal Large Language Models have shown prominent understanding and reasoning abilities
across diverse vision-language tasks [ 35,50,55]. Beyond the LLM backbone, MLLMs incorporate
two key components: a vision encoder and a vision-language integration module. The former typically
uses a pre-trained visual encoder [ 14], while the latter varies significantly in design, like MLP-based
projectors [ 35], Perceiver [ 1], and Q-Former [ 13]. While MLLMs excel at processing human queries
and interpreting visual context, they remain prone to knowledge gaps and hallucinations. Although
this issue is inherent to all LLMs, it is more pronounced in MLLMs due to the limited availability of
high-quality, large-scale multimodal data.
While traditional VQA [ 2,19] benchmarks evaluate vision-language understanding primarily within
the visual context, knowledge-intensive VQA significantly increases the challenge by requiring
specific or detailed knowledge beyond the image content. Early benchmarks such as OK-VQA [ 39]
and A-OKVQA [ 48] highlight the importance of commonsense knowledge in VQA, which can be
effectively addressed by MLLMs trained on large and diverse corpora. However, E-VQA [ 40] and
InfoSeek [ 6] introduced greater challenges by encompassing a broad range of Wikipedia entities
and requiring fine-grained knowledge about them. Consequently, modern MLLMs often fail to
answer such questions accurately, as the relevant knowledge is either missing or represents long-tail
distributions in their training data.
RAG is commonly used in LLMs to tackle issues such as outdated information and hallucina-
tions [ 16,41]. By dynamically combining external knowledge with the model’s built-in abilities,
RAG offers an efficient approach for tasks requiring extensive knowledge. Inspired by this, RA-
VQA [ 31], Wiki-LLaV A [ 4], and EchoSight [ 53] have successfully applied retrieval augmentation
to knowledge-intensive VQA, but their retrieval suffers from the modality gap [ 29] between multi-
modal queries and textual knowledge bases. Recent studies [ 11,56] leveraged MLLMs to identify
relevant information from retrieved passages, which depend on multiple calls to MLLMs and lead to
substantially higher inference overhead. Moreover, existing RAG-based methods typically retrieve
3

In which park is ...Top K
Passages
Visual
EncoderzIn which park is the mountain in this
picture located?QuestionEntity: Half Dome,  ...
Relation : (Half Dome,  Yosemite Valle, ...)
Chunks : Half Dome is a quartz ....Entity &  RelationMLLM
Answer  Generation Embedding-based Retrieval Graph-based RetrievalTextual & Visual Graph Extraction
Entity Matching Relation MatchingMultimodal Graph Generation
ascendoverviewpoint
reachHalf Dome
Yosemite
ValleyGlacier 
Point
George
Anderson
eastern
end
located
inRainbow
Hiker
Yosemite
National  Park
Subgraph 
Merging
(Online)Passage #1
Passage #3Passage #2
Multimodal KG
Extraction
(Offline)incloud
skyover inrainbow
belong-
ing to grassbelonging to
treelake
behind
forest
behind
behindmountain
field
Entity
Retrieval
Relationship
Retrieval
Entity: Half Dome   
Description:  Half Dome is a quartz monzonite
batholith at the eastern ...
Relation: (Half Dome,  Yosemite Valley )
Description:  Half Dome is located at the
eastern end of Yosemite Valley  ...Relation: (Half Dome,  Yosemite Valley )
Description:  Half Dome is located at the
eastern end of Yosemite Valley  ...Relation: (Half Dome,  Yosemite Valley )
Description:  Half Dome is located at the
eastern end of Yosemite Valley  ...Entity: Half Dome   
Description:  Half Dome is a quartz monzonite
batholith at the eastern ...Entity: Half Dome   
Description:  Half Dome is a quartz monzonite
batholith at the eastern ...
Multimodal QueryKnowledge BaseMultimodal Document
All Images
Figure 2: An overview of our mKG-RAG consists of a multimodal knowledge graph construction
pipeline ( top) and a dual-stage retrieval paradigm for answer generation ( bottom ).
unstructured documents, overlooking both the noise present in retrieval sources and the logical
relationships among knowledge elements. This results in noisy and disorganized knowledge that
increases the reasoning burden on MLLMs. To address this challenge, recent research has begun
exploring the use of Knowledge Graphs (KGs), which provide structured representations of entities
and their relationships, to enhance the generative capabilities of LLMs [ 22,27,15,59,38]. However,
these efforts primarily focus on textual KGs, leaving the potential of multimodal KGs largely underex-
plored. To bridge this gap, our work is the first to integrate multimodal KGs into the RAG framework,
specifically designed for vision-language tasks that require fine-grained external knowledge.
3 The Proposed Method: mKG-RAG
In the knowledge-based VQA task, the model receives an image-question pair (Iq, q)as input and
is required to generate a textual answer a, potentially leveraging an accessible knowledge base B
as additional context. In our setting, the knowledge source is composed of multimodal documents
featuring both text articles Tand their corresponding image assets I,i.e.,B={(Ti, Ii)}N
i=1. The
core objectives of our multimodal retrieval-augmented generation framework are twofold: (1) to
effectively convert the unstructured knowledge base Binto structured multimodal KGs, and (2) to
precisely retrieve query-relevant knowledge from multimodal KGs while capturing the underlying
structural relationships, thereby augmenting the knowledge scope of MLLMs.
The visual workflow of the proposed mKG-RAG is depicted in Figure 2, showcasing two key
innovations. First, a multimodal knowledge graph construction pipeline is introduced, leveraging
MLLMs to convert plain multimodal documents into structured knowledge representations, i.e.,
graphs. Then, a dual-stage retrieval paradigm is proposed to perform fine-grained graph retrieval over
a query-specific multimodal KG constructed from subgraphs of documents initially retrieved through
coarse-grained vector search.
3.1 Multimodal Knowledge Graph Construction
Existing retrieval-augmented VQA models struggle with noisy context and overlook structural
relationships due to retrieving fragmented textual chunks. A promising solution is to retrieve from
structured knowledge sources, such as knowledge graphs. Nevertheless, the off-the-shelf multimodal
KGs [ 36] are typically designed for common entities, which is unsuitable for addressing VQA
cases involving detailed or long-tail knowledge, not to mention the domain-specific or even private
4

knowledge. Thus, this work explores an effective multimodal KG construction pipeline to extract
semantic-consistent and modality-aligned entities and relationships from accessible multimodal
documents for the task of knowledge-based VQA. Specifically, for each document (T, I)∈ B, where
article T={t1, ..., t n}typically contains multiple sections and I={i1, ..., i m}is a set of images,
we first segment it into manageable pieces. Sections without images are split or merged based on a
fixed chunk size [ 16], while sections containing images are preserved in their entirety to maintain
alignment between images and text. As illustrated in Figure 2, each segment is then processed by
three key modules. Textual Graph Extraction identifies entities and their relationships from text,
while Visual Graph Extraction detects prominent objects and their interactions from images. Finally,
the Multimodal Graph Generation module fuses the textual and visual entities and relations into a
unified multimodal graph.
Textual Graph Extraction. Following prior work [ 20], we process each textual piece by prompting
LLMs to identify key entities (nodes) and meaningful relationships (edges), thereby forming a textual
subgraph Gt= (N,E). Like the example in Figure 2, each entity niinNcontains a unique name
and a detailed description, offering an abstract representation to facilitate subsequent retrieval. Each
relationship eij∈ Econnects head and tail entities (ni, nj)and includes a concise relation summary.
Visual Graph Extraction. The textual subgraph has distilled the skeleton of textual chunks, including
informative entities and relationships, but it lacks visual elements, a critical component in VQA tasks.
A naive strategy is to supply Gtwith corresponding images [ 36] directly. However, considering that
images often contain multiple objects and background noise, we propose augmenting the textual
subgraph with fine-grained region information. Each region possibly represents an individual entity or
a relationship among two or more entities, as depicted in Figure 2. For simplicity, this work focuses
exclusively on binary relationships, leaving the study of hyper-relationships [ 37] for future research.
Specifically, we employ the Scene Graph Generation (SGG) techniques [ 25] to extract a precise
visual graph for each image in I. The visual graph is formalized as Gv= (V,R), where V={vi}Nv
i=1
represents the set of visual objects with predicted category labels and bounding boxes, and R=
{rij}i̸=jdenotes the visual relationships between objects. Unlike object detection [ 47], SGG offers
additional relational information, facilitating efficient vision-text relationship matching.
Multimodal Graph Generation. At the core of the construction pipeline lies the challenge of
merging the textual and visual graphs into a semantic-consistent and modality-aligned multimodal
graph. Directly matching textual and visual entities/relationships based on image-text similarity [ 46]
is limited to shallow or global alignment, lacking the capacity to capture fine-grained and contextual
correspondences. Given the impressive vision-language understanding abilities of MLLMs [ 35], a
promising solution is to employ MLLMs as the vision-text matcher, effectively aligning semantically
consistent visual and textual entities/relationships. Thus, the following prompt is designed:
Vision-Text Matching Prompt : <Prefix Instruction> <IMAGE> [Textual Entities & Relationships]
[Visual Entities & Relationships]
Here, <Prefix Instruction> explains the input format of textual and visual graphs and guides the
MLLMs in matching their entities and relationships. <IMAGE> denotes the corresponding image of
the visual graph and contains only the original image, without extra regions. To enable MLLMs to
comprehend graph structures, we convert both GtandGvinto natural language format. For Gt, each
entity and relationship is expressed using its name and associated description, formatted as a sentence.
The visual objects and relationships in Gvare encoded as “<Object-ID>: <category>, <bbox>”
and“<Relation-ID>: <subject>, <relation>, <object>” , respectively. Importantly, visual entities
include only the predicted category and normalized bounding box, from which MLLMs can locate
the corresponding region within <IMAGE> , without requiring actual regional images [ 54]. This
design enables efficient inference by allowing simultaneous processing of all objects and relations
inGv. To ensure MLLMs follow the prefix instruction and produce the desired output, we further
enhance their reasoning ability by providing several high-quality exemplars. Detailed prompts are
provided in Appendix.A .
The whole process of vision-text matching is denoted as:
M={(n, v)i}Ne
i=1∪ {(e, r)j}Nr
j=1=Fmllm(I,Gt,Gv). (1)
Here,Mdenotes a set comprising Nematched entities and Nrmatched relationships. As depicted in
Figure 2, the image region of v(r) is attached as an attribute to its corresponding textual counterpart
5

n(e). Since a visual relationship rinvolves two object regions, we merge them using the union of
their bounding boxes.
Through the above steps, we generate an image-text-aligned multimodal subgraph Gfor each doc-
ument segment. These subgraphs are then aggregated into a complete graph by merging identical
nodes and edges. Notably, only subgraphs from the same document are merged, ensuring that each
document produces an independent multimodal KG. During retrieval, relevant KGs from different
documents are dynamically composed based on the retrieval results. Since the construction process is
query-independent, the entire pipeline can be executed offline, and each document requires processing
only once.
3.2 Dual-stage Retrieval Paradigm
To unleash the potential of constructed multimodal KGs, we further introduce a dual-stage retrieval
framework inspired by human cognitive processes. When encountering unfamiliar multimodal queries,
humans typically: (1) filter relevant supporting evidence from vast external multimodal sources, then
(2) analyze and organize the extracted information into coherent structures for reasoning [ 59]. Our
framework accordingly implements a coarse-grained vector similarity search followed by fine-grained
graph retrieval.
Embedding-based Retrieval. For large-scale knowledge bases containing millions of passages,
direct graph retrieval is inefficient, as each passage may include hundreds of nodes and edges, greatly
expanding the search space. Thus, we first perform coarse-grained recall using vector search to
identify candidates. Given a query (Iq, q)and a set of multimodal articles {(Ti, Ii)}N
i=1, a similarity
matrix Scan be obtained:
S={si=⟨Eq(Iq, q)· Ee(Ii, Ti)⟩, i= 1, ..., N}, (2)
where ⟨·⟩denotes the cosine similarity; EqandEeare multimodal encoders designed for query
and evidence, respectively, as shown in Figure 3. Based on matrix S, the top Kdhighest-scoring
documents are collected.
Statement Visual
Encoder Q-Former🔥
❄
 Question
Convertor 
 Rephrasing
Question 🔥...
...
Query Tokens ... ...Evidence 
Figure 3: Architecture Design of Question-aware
Multimodal Retriever.Graph-based Retrieval. Previous methods re-
trieve text chunks directly from candidate doc-
uments [53], which often introduces contextual
noise and impairs reasoning performance. In
contrast, our approach performs graph-based
retrieval to identify query-relevant entities and
relationships. These entities and relationships
serve as distilled knowledge representations, sig-
nificantly reducing noise and enabling more ac-
curate retrieval.
Specifically, a query-specific multimodal graph
Gmis constructed by merging the offline-
generated subgraphs corresponding to the can-
didate documents retrieved in the first stage. By
limiting the merge to only relevant documents,
such an online strategy effectively reduces am-
biguous entities and relationships that often arise from cross-document knowledge inconsisten-
cies [ 15]. Next, query-relevant entities and relationships are identified by computing embedding
similarities between the multimodal query and each entity/relationship in Gm. The embedding vector
of the given entity and relationship can be formalized as fe=Ee(n, v)andfr=Ee(e, r). Here,
topKgbest-matched candidates will be selected, e.g., the entity a1and the relationship (b2, b4)in
Figure 2. Combining Kgmatched entities or relationships, we get a relevant subgraph G0
r. However,
similarity-based retrieval alone may yield incomplete information, potentially omitting critical evi-
dence to answer the question entirely. To this end, we leverage the inherent structural properties of
the graph and expand Grby incorporating information from its l-hop neighbors, i.e.,
Gl
r=Graph Traversal (Gm,G0
r, l), (3)
where Graph Traversal is implemented by the breadth-first search. Notably, we selectively incorporate
only query-relevant neighbors, as shown by the green nodes in Figure 2.
6

Table 1: Retrieval performance on E-VQA set
ModelRet.
ModeE-VQA
R@1 R@5 R@10 R@20 R@50
Nomic-text T →T 2.0 4.1 5.6 7.8 11.1
Nomic-vision V →V 9.3 23.0 29.3 36.0 45.6
CLIP ViT-L/14 T →T 2.0 4.7 6.4 8.8 12.1
CLIP ViT-L/14 V →V 11.2 28.5 36.2 44.1 54.8
CLIP ViT-L/14 T →V 1.1 3.1 4.6 7.3 12.3
CLIP ViT-L/14 V →T 3.8 10.2 13.6 18.0 23.9
QM-Retriever MM 18.9 36.8 46.2 55.6 66.7Table 2: Retrieval performance on InfoSeek set
ModelRet.
ModeInfoSeek
R@1 R@5 R@10 R@20 R@50
Nomic-text T →T 11.0 19.3 24.2 30.4 40.6
Nomic-vision V →V 35.0 56.5 63.3 69.3 75.5
CLIP ViT-L/14 T →T 9.2 15.8 19.3 23.3 30.0
CLIP ViT-L/14 V →V 40.0 63.4 70.9 77.7 83.7
CLIP ViT-L/14 T →V 8.5 18.8 24.6 31.7 42.5
CLIP ViT-L/14 V →T 20.1 40.1 49.2 58.3 68.9
QM-Retriever MM 49.7 71.6 78.0 82.5 89.1
The retrieved context comprises both graph elements (entities and relationships) and their associated
textual segments. The former provides a structured knowledge outline, while the latter supplies the
contextual details. Finally, the concatenated image, question, and context are fed into MLLMs for
answer generation.
Question-aware Multimodal Retriever. Standard multimodal retrievers are optimized for semantic
similarity rather than question relevance, often failing to retrieve precise evidence needed for answer
generation, even if the returned content is semantically related. To address this issue, this work
proposes a Question-aware Multimodal Retriever (QM-Retriever) targeting evidence retrieval for
VQA tasks.
As shown in Figure 3, the retriever is adapted from the Q-Former [ 28] by incorporating an additional
Visual Encoder Fvand a Question Converter Fq. We employ the BLIP-2 [ 28] pre-trained vision
encoder as Fvto extract image features. The Question Converter reformulates interrogative questions
into declarative forms to address grammatical mismatches with evidence texts, which could otherwise
hinder retrieval accuracy. Importantly, the reformulation process occurs in the latent space rather than
the language space. Given an image-question pair (Iq, q), QM-Retriever encodes it into a fix-sized
embedding Zq:
Zq=Q-Former (Z,Fv(Iq),Fq(q)), (4)
where Zis a set of learnable tokens introduced by Q-Former. The resulting embedding Zqcan then
be used for vector-based retrieval. Note that the QM-Retriever omits the Question Converter when
operating as the evidence encoder.
To optimize the QM-Retriever, a query-evidence dataset is built based on the training set of E-
VQA [ 40], where each multimodal query (Iq, q)is paired with its corresponding ground-truth
evidence (Ie, Te). Here, Terepresents evidence text, and Ierefers to the associated image from the
evidence section. For sections without visual content, black images are used as placeholders.
The optimization of our QM-Retriever involves two key objectives: (1) Question-Evidence Alignment.
To retrieve query-relevant evidence, we employ contrastive learning [ 21,5] to align the features
of multimodal query and evidence by encouraging positive query-evidence pairs to have similar
representations in contrast to the negative pairs in a batch, i.e.,
Lcon=−logexp( sim(Zq, Ze)/τ)
ΣB
k=1exp( sim(Zq, Zk)/τ). (5)
Here, Bdenotes the batch size, and τis a temperature parameter. (2) Question Reformulation. We
leverage LLMs to convert the original question qinto a declarative statement sthat emphasizes the
scene context. By encoding (Iq, s)with QM-Retriever, we obtain a declarative representation Zsas
a reference. Then, Kullback-Leibler divergence is measured to minimize the divergence between
the distributions of ZqandZs. Finally, the total objective is formulated as the linear combination
controlled by a hyperparameter:
L=Lcon+α DKL(p(Zq|Iq, q)∥p((Zs|Iq, s))). (6)
Notably, the Q-Former is initialized with BLIP-2’s weights and fine-tuned jointly with Fq, while Fv
remains frozen.
7

Table 3: Main results of models with external knowledge on the E-VQA and InfoSeek datasets. *
denotes that the model is further fine-tuned on the corresponding dataset. †and‡represent the
variant of our mKG-RAG with different retrievers.
Model LLM / MLLMRetrieval Mode E-VQA InfoSeek
Retriever Text Image Single-Hop All Unseen-Q Unseen-E All
Zero-shot MLLMs
BLIP-2 [28] Flan-T5XL – 12.6 12.4 12.7 12.3 12.5
InstructBLIP [13] Flan-T5XL – 11.9 12.0 8.9 7.4 8.1
LLaV A-v1.5 [34] Vicuna-7B – 16.3 16.9 9.6 9.4 9.5
LLaV A-More [10] LLaMA-3.1-8B – 15.8 16.0 9.0 8.2 8.6
Qwen2-VL [50] Qwen2-VL-7B – 19.9 19.7 19.8 18.5 19.2
Retrieval-Augmented Models
RORA-VLM [44] Vicuna-7B CLIP + GS – 20.3 25.1 27.3 –
Wiki-LLaV A*[4] Vicuna-7B CLIP ViT-L/14 21.8 26.4 30.1 27.8 28.9
EchoSight [53] LLaMA-3.1-8B EV A-CLIP-8B 22.4 21.7 30.0 30.7 30.4
EchoSight [53] LLaMA-3.1-8B EV A-CLIP-8B 26.4 24.9 18.0 19.8 18.8
mR2AG*[56] Vicuna-7B CLIP ViT-L/14 – – 40.6 39.8 40.2
ReflectiV A*[11] LLaMA-3.1-8B EV A-CLIP-8B 28.0 29.2 40.4 39.8 40.1
ReflectiV A*[11] LLaMA-3.1-8B EV A-CLIP-8B 35.5 35.5 28.6 28.1 28.3
Graph Retrieval-Augmented Models
mKG-RAG† LLaMA-3.1-8B CLIP ViT-L/14 24.4 23.4 24.1 22.3 23.2
mKG-RAG‡ LLaMA-3.1-8B CLIP ViT-L/14 24.6 23.7 21.3 19.8 20.6
mKG-RAG LLaMA-3.1-8B QM-Retriever 27.1 26.1 32.9 31.3 32.1
mKG-RAG*† LLaMA-3.1-8B CLIP ViT-L/14 36.6 34.9 29.8 28.5 29.1
mKG-RAG*‡ LLaMA-3.1-8B CLIP ViT-L/14 32.9 31.0 29.4 27.3 28.3
mKG-RAG*LLaMA-3.1-8B QM-Retriever 38.4 36.3 41.4 39.6 40.5
4 Experiments
4.1 Experimental Setup
Datasets and Knowledge Base. Our method is evaluated on E-VQA [40] and InfoSeek [6], which
contain question-answer pairs linked to documents from Wikipedia. E-VQA offers a knowledge
base comprising 2M Wikipedia pages, where each question-answer pair is annotated with supporting
Wikipedia articles, relevant evidence paragraphs, and associated images. For InfoSeek, since there is
no publicly released knowledge base, we utilize a subset of 100K documents from E-VQA filtered by
EchoSight [53] as our knowledge source.
Implementation Details. We use the Llama-3.2-11B-Vision model as the MLLM for multimodal
KG Construction, including textual entity-relationship recognition and vision-text matching. A
lightweight one-stage SGG model, EGTR [ 25], is applied to produce scene graphs for images in
the knowledge base. In the first-stage retrieval, we utilize the FAISS [ 26] for efficient approximate
nearest neighbor search and select the top 10 ( Kd) best-matched documents. For the graph retrieval,
we empirically set the Kgandlto10and1, respectively. Unless otherwise indicated, we adopt
LLaV A-More [ 10] as the multimodal answer generator, following the setup in ReflectiV A [ 11]. More
details are provided in the Appendix.B .
4.2 Performance Comparison
Results on Retrieval. To assess the effectiveness of multimodal retrieval using QM-Retriever, we
conduct comparative analyses against unimodal and cross-modal retrievers in selecting the most
relevant documents for VQA queries. Specifically, we use Nomic-Embed-v1.5 [ 43] and CLIP ViT-
L/14@336 [ 46] as retrieval baselines and examine four feasible retrieval combinations: text-to-text
(T→T), vision-to-vision (V →V), text-to-vision (T →V), and vision-to-text (V →T).
Table 1 and Table 2 report the Recall scores on E-VQA and InfoSeek, respectively. QM-Retriever
consistently outperforms all baseline methods, achieving average improvements of 9.9% (E-VQA)
and 7.0% (InfoSeek) over the second-best approach. The strong recall performance ensures that
mKG-RAG operates on highly relevant knowledge graphs constructed in the fine-grained retrieval
phase, as further supported by our ablation studies. Additionally, the results reveal that V →V retrieval
consistently beats other unimodal and cross-modal configurations, underscoring the critical role of
visual content in VQA tasks.
8

Table 4: VQA accuracy on E-VQA across different MLLM architectures with varying sizes
MLLM E-VQAInternVL3 LLaMA-3.2 LLaVA-v1.5 DeepSeek-VL2 Qwen2.5-VL
8B 11B 7B 13B 3B 16B 3B 7B 32B
Zero-shotSingle-Hop 22.4 27.0 15.8 16.1 22.0 22.4 19.1 21.0 27.1
All 23.0 28.9 16.2 16.6 21.6 22.3 18.9 20.8 27.3
mKG-RAGSingle-Hop32.7
↑10.337.2
↑10.225.0
↑9.227.7
↑11.628.4
↑6.431.1
↑8.728.9
↑9.830.4
↑9.436.5
↑9.4
All32.7
↑9.738.5
↑9.624.6
↑8.427.8
↑11.227.4
↑5.829.9
↑7.628.2
↑9.329.6
↑8.836.5
↑9.2
Qwen 2-VL： 
Indonesian
GPT -4o:
Indo-Saracenic
mKG -RAG:
Javanese
Q: Who designed this museum?Qwen 2-VL:
Jozef KUBA
GPT -4o:
Matej Walch
mKG -RAG:
J. Langer 
Qwen 2-VL:
air space
GPT -4o:
a ring of dark spots
mKG -RAG:
brownish  spotsQwen 2-VL:
PARK
GPT -4o:
New York City Department 
of Parks and Recreation
mKG -RAG:
Abingdon Square ConservancyQwen 2-VL:
400
GPT -4o:
I can’t answer without 
specific details.
mKG -RAG:
250-metre
Q: What is the architectural style of 
this mosque?Q: Who is in charge of maintaining this park?Q: How long is the track at this velodrome?
Q: What forms a ring on the larger end 
of the egg of this bird?
Qwen 2-VL:
1000000
GPT -4o:
26.04 km2
mKG -RAG:
3.82 km2
Q: What is the surface area of this lake?
Figure 4: Qualitative results of Qwen2-VL-7B, GPT-4o and mKG-RAG on E-VQA dataset.
Results on E-VQA and InfoSeek. In this section, we compare mKG-RAG with Zero-shot MLLMs
and RAG-based approaches on the benchmarks mentioned above. The results in Table 3 demonstrate
that zero-shot MLLMs struggle with knowledge-based VQA tasks, particularly on the InfoSeek
dataset. These limitations underscore the critical need for external knowledge integration. By
augmenting LLaV A-More with mKG-RAG, we achieve substantial improvements, over 20.3% on
E-VQA and 31.9% on InfoSeek, highlighting the value of retrieval augmentation.
Furthermore, our method achieves state-of-the-art performance on both datasets. Under the fine-
tuning setting, mKG-RAG*surpasses both mR2AG*and ReflectiV A*. Even without fine-tuning,
mKG-RAG outperforms EchoSight by 1.2% and 1.7%, respectively. These results highlight the
advantages of integrating RAG with multimodal KGs and demonstrate the effectiveness of our QM-
Retriever. Table 3 also includes two mKG-RAG variants that replace QM-Retriever with text-only
and vision-only CLIP for entity/relationship retrieval, while still using documents retrieved by QM-
Retriever to construct multimodal KGs. In the text-only variant, both questions and image captions
are used as queries to provide more context, explaining its better performance over the vision-only
version. However, they both remain less effective than our full approach with QM-Retriever.
Consistency across Architectures. In Table 4, we provide a detailed comparison of VQA scores
across MLLMs of varying parameter sizes, including InternVL3 [ 58], LLaMA-3.2-Vision2, LLaV A-
v1.5 [ 34], DeepSeek-VL2 [ 52], and Qwen2.5-VL [ 3]. When enhanced with our mKG-RAG frame-
work, these models achieve average performance gains of 9.4% on single-hop queries and 8.7% on
overall scenarios, demonstrating the method’s strong generalization across different architectures and
scales.
Qualitative Results. Figure 4 shows a qualitative comparison of mKG-RAG with zero-shot Qwen2-
VL and GPT-4o. While the latter two tend to produce plausible but incorrect or evasive responses,
mKG-RAG consistently handles knowledge-intensive queries, especially those involving precise
numerical and temporal reasoning.
2https://huggingface.co/meta-llama/Llama-3.2-11B-Vision
9

Table 5: The ablation study of the design of mKG-RAG
MethodE-VQA InfoSeek
Single-Hop All Un-Q Un-E All
mKG-RAG 38.4 36.3 41.4 39.6 40.5
w/o QM-Retriever 34.2 31.6 38.9 37.9 38.4
w/o Graph Retrieval 30.1 28.2 33.3 32.7 33.0
w/o Graph Expansion 37.2 35.0 40.8 39.4 40.1
Table 6: The ablation study of how the retrieval number of entities/relationships affects the VQA
accuracy on E-VQA.
Model Ret. Mode Kg= 1 Kg= 5 Kg= 10 Kg= 20
mKG-RAG† Textual 29.1 33.9 34.9 35.9
mKG-RAG‡ Visual 23.0 29.6 31.0 32.0
mKG-RAG Multimodal 29.2 35.1 36.3 36.9
4.3 Ablation Study
Impact of Coarse-grained Retrieval. To quantify the impact of coarse-grained document retrieval,
we conduct an ablation experiment replacing QM-Retriever with visual-only CLIP (ViT-L/14@336)
for top- Kddocument selection. The results in Table 5 reveal significant performance drops: overall
VQA accuracy of mKG-RAG decreases by 4.7% on E-VQA and 2.1% on InfoSeek. This ablation
conclusively demonstrates the critical role of first-stage retrieval and QM-Retriever’s superiority over
unimodal alternatives.
Effectiveness of Graph-based Retrieval. In our method, the entities and relationships extracted from
documents form a distilled knowledge graph, reducing noise and enabling more effective retrieval
than direct text chunk matching. To validate this insight, we replace graph-based retrieval with a naive
chunk-based alternative. Specifically, we segment retrieved documents into fixed-size chunks and
select those relevant to the given question and image description. As shown in Table 5, chunk-based
retrieval leads to a substantial accuracy drop, 8.1% on E-VQA and 7.5% on InfoSeek.
Contribution of Graph Expansion. mKG-RAG enhances the constructed subgraph through l-hop
neighbor expansion, effectively capturing potentially missing but relevant knowledge connections.
Table 5 shows that omitting graph expansion leads to consistent performance drops of 1.3% (E-VQA)
and 0.4% (InfoSeek), demonstrating its critical contribution to our mKG-RAG.
Impact of Varying Retrieval Number. In Table 6, we further analyze the impact of Kg, the
number of retrieved entities and relationships, on our method. As Kgincreases from 1 to 20, the
overall accuracy of mKG-RAG and its variants gradually improves, as higher recall rates enhance
the likelihood of capturing relevant knowledge. However, when Kg>10, the benefit diminishes
due to longer contexts and more noise. Thus, setting Kg= 10 offers a practical trade-off. Notably,
mKG-RAG still performs competitively even at Kg= 1, thanks to its graph expansion strategy,
which enables the model to gather additional relevant information.
5 Conclusion
We propose mKG-RAG, a novel retrieval-augmented generation framework that integrates multimodal
knowledge graphs (KGs) to overcome the knowledge limitations of multimodal large language models
(MLLMs). Our framework constructs structured, modality-aligned KGs using MLLM-driven keyword
extraction and cross-modal alignment, and employs a dual-stage retrieval system, which combines
vector-based and graph-based retrieval for precise knowledge augmentation. Extensive experiments
show that mKG-RAG outperforms state-of-the-art methods, with ablation studies validating each
component’s contributions.
10

References
[1]Alayrac Jean-Baptiste, Donahue Jeff, Luc Pauline, Miech Antoine, Barr Iain, Hasson Yana,
Lenc Karel, Mensch Arthur, Millican Katherine, Reynolds Malcolm, others . Flamingo: a visual
language model for few-shot learning // Advances in neural information processing systems.
2022. 35. 23716–23736.
[2]Antol Stanislaw, Agrawal Aishwarya, Lu Jiasen, Mitchell Margaret, Batra Dhruv, Zitnick
C Lawrence, Parikh Devi . Vqa: Visual question answering // Proceedings of the IEEE
international conference on computer vision. 2015. 2425–2433.
[3]Bai Shuai, Chen Keqin, Liu Xuejing, Wang Jialin, Ge Wenbin, Song Sibo, Dang Kai, Wang Peng,
Wang Shijie, Tang Jun, Zhong Humen, Zhu Yuanzhi, Yang Mingkun, Li Zhaohai, Wan Jianqiang,
Wang Pengfei, Ding Wei, Fu Zheren, Xu Yiheng, Ye Jiabo, Zhang Xi, Xie Tianbao, Cheng Zesen,
Zhang Hang, Yang Zhibo, Xu Haiyang, Lin Junyang . Qwen2.5-VL Technical Report. 2025.
[4]Caffagni Davide, Cocchi Federico, Moratelli Nicholas, Sarto Sara, Cornia Marcella, Baraldi
Lorenzo, Cucchiara Rita . Wiki-llava: Hierarchical retrieval-augmented generation for mul-
timodal llms // Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. 2024. 1818–1826.
[5]Chen Ting, Kornblith Simon, Norouzi Mohammad, Hinton Geoffrey . A simple framework for
contrastive learning of visual representations // International conference on machine learning.
2020. 1597–1607.
[6]Chen Yang, Hu Hexiang, Luan Yi, Sun Haitian, Changpinyo Soravit, Ritter Alan, Chang
Ming-Wei . Can Pre-trained Vision and Language Models Answer Visual Information-Seeking
Questions? // Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing. 2023. 14948–14968.
[7]Chen Zhe, Wu Jiannan, Wang Wenhai, Su Weijie, Chen Guo, Xing Sen, Zhong Muyan, Zhang
Qinglong, Zhu Xizhou, Lu Lewei, others . Internvl: Scaling up vision foundation models and
aligning for generic visual-linguistic tasks // Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition. 2024. 24185–24198.
[8]Chiang Wei-Lin, Li Zhuohan, Lin Ziqing, Sheng Ying, Wu Zhanghao, Zhang Hao, Zheng Lianmin,
Zhuang Siyuan, Zhuang Yonghao, Gonzalez Joseph E, others . Vicuna: An open-source chatbot
impressing gpt-4 with 90%* chatgpt quality // See https://vicuna. lmsys. org (accessed 14 April
2023). 2023. 2, 3. 6.
[9]Chung Hyung Won, Hou Le, Longpre Shayne, Zoph Barret, Tay Yi, Fedus William, Li Yunxuan,
Wang Xuezhi, Dehghani Mostafa, Brahma Siddhartha, others . Scaling instruction-finetuned
language models // Journal of Machine Learning Research. 2024. 25, 70. 1–53.
[10] Cocchi Federico, Moratelli Nicholas, Caffagni Davide, Sarto Sara, Baraldi Lorenzo, Cor-
nia Marcella, Cucchiara Rita . LLaV A-MORE: A Comparative Study of LLMs and Visual
Backbones for Enhanced Visual Instruction Tuning // arXiv preprint arXiv:2503.15621. 2025.
[11] Cocchi Federico, Moratelli Nicholas, Cornia Marcella, Baraldi Lorenzo, Cucchiara Rita . Aug-
menting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question
Answering // Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. 2025.
[12] Cui Can, Ma Yunsheng, Cao Xu, Ye Wenqian, Zhou Yang, Liang Kaizhao, Chen Jintai,
Lu Juanwu, Yang Zichong, Liao Kuei-Da, others . A survey on multimodal large language mod-
els for autonomous driving // Proceedings of the IEEE/CVF winter conference on applications
of computer vision. 2024. 958–979.
[13] Dai Wenliang, Li Junnan, Li Dongxu, Tiong Anthony, Zhao Junqi, Wang Weisheng, Li Boyang,
Fung Pascale, Hoi Steven . InstructBLIP: Towards General-purpose Vision-Language Models
with Instruction Tuning // Thirty-seventh Conference on Neural Information Processing Systems.
2023.
11

[14] Dosovitskiy Alexey, Beyer Lucas, Kolesnikov Alexander, Weissenborn Dirk, Zhai Xiaohua,
Unterthiner Thomas, Dehghani Mostafa, Minderer Matthias, Heigold Georg, Gelly Sylvain,
Uszkoreit Jakob, Houlsby Neil . An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale // International Conference on Learning Representations. 2021.
[15] Edge Darren, Trinh Ha, Cheng Newman, Bradley Joshua, Chao Alex, Mody Apurva, Truitt
Steven, Metropolitansky Dasha, Ness Robert Osazuwa, Larson Jonathan . From local to global:
A graph rag approach to query-focused summarization // arXiv preprint arXiv:2404.16130.
2024.
[16] Fan Wenqi, Ding Yujuan, Ning Liangbo, Wang Shijie, Li Hengyun, Yin Dawei, Chua Tat-Seng,
Li Qing . A survey on rag meeting llms: Towards retrieval-augmented large language models //
Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.
2024. 6491–6501.
[17] Fan Wenqi, Wang Shijie, Huang Jiani, Chen Zhikai, Song Yu, Tang Wenzhuo, Mao Haitao, Liu
Hui, Liu Xiaorui, Yin Dawei, others . Graph machine learning in the era of large language
models (llms) // arXiv preprint arXiv:2404.14928. 2024.
[18] Fan Wenqi, Zhou Yi, Wang Shijie, Yan Yuyao, Liu Hui, Zhao Qian, Song Le, Li Qing . Com-
putational Protein Science in the Era of Large Language Models (LLMs) // arXiv preprint
arXiv:2501.10282. 2025.
[19] Goyal Yash, Khot Tejas, Summers-Stay Douglas, Batra Dhruv, Parikh Devi . Making the v in vqa
matter: Elevating the role of image understanding in visual question answering // Proceedings
of the IEEE conference on computer vision and pattern recognition. 2017. 6904–6913.
[20] Guo Zirui, Xia Lianghao, Yu Yanhua, Ao Tu, Huang Chao . LightRAG: Simple and Fast
Retrieval-Augmented Generation // arXiv preprint arXiv:2410.05779. 2024.
[21] He Kaiming, Fan Haoqi, Wu Yuxin, Xie Saining, Girshick Ross . Momentum contrast for
unsupervised visual representation learning // Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition. 2020. 9729–9738.
[22] He Xiaoxin, Tian Yijun, Sun Yifei, Chawla Nitesh, Laurent Thomas, LeCun Yann, Bresson Xavier,
Hooi Bryan . G-retriever: Retrieval-augmented generation for textual graph understanding
and question answering // Advances in Neural Information Processing Systems. 37. 2024.
132876–132907.
[23] Hogan Aidan, Blomqvist Eva, Cochez Michael, d’Amato Claudia, Melo Gerard De, Gutierrez
Claudio, Kirrane Sabrina, Gayo José Emilio Labra, Navigli Roberto, Neumaier Sebastian,
others . Knowledge graphs // ACM Computing Surveys (Csur). 2021. 54, 4. 1–37.
[24] Hu Edward J, Shen Yelong, Wallis Phillip, Allen-Zhu Zeyuan, Li Yuanzhi, Wang Shean, Wang
Lu, Chen Weizhu, others . Lora: Low-rank adaptation of large language models. // ICLR. 2022.
1, 2. 3.
[25] Im Jinbae, Nam JeongYeon, Park Nokyung, Lee Hyungmin, Park Seunghyun . Egtr: Extracting
graph from transformer for scene graph generation // Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. 2024. 24229–24238.
[26] Johnson Jeff, Douze Matthijs, Jégou Hervé . Billion-scale similarity search with GPUs // IEEE
Transactions on Big Data. 2019. 7, 3. 535–547.
[27] LUO LINHAO, Li Yuan-Fang, Haffari Gholamreza, Pan Shirui . Reasoning on Graphs: Faithful
and Interpretable Large Language Model Reasoning // The Twelfth International Conference on
Learning Representations. 2024.
[28] Li Junnan, Li Dongxu, Savarese Silvio, Hoi Steven . Blip-2: Bootstrapping language-image
pre-training with frozen image encoders and large language models // International conference
on machine learning. 2023. 19730–19742.
12

[29] Liang Victor Weixin, Zhang Yuhui, Kwon Yongchan, Yeung Serena, Zou James Y . Mind the gap:
Understanding the modality gap in multi-modal contrastive representation learning // Advances
in Neural Information Processing Systems. 2022. 35. 17612–17625.
[30] Lin Ji, Yin Hongxu, Ping Wei, Molchanov Pavlo, Shoeybi Mohammad, Han Song . Vila: On pre-
training for visual language models // Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition. 2024. 26689–26699.
[31] Lin Weizhe, Byrne Bill . Retrieval Augmented Visual Question Answering with Outside Knowl-
edge // Proceedings of the 2022 Conference on Empirical Methods in Natural Language
Processing. 2022. 11238–11254.
[32] Lin Weizhe, Chen Jinghong, Mei Jingbiao, Coca Alexandru, Byrne Bill . Fine-grained late-
interaction multi-modal retrieval for retrieval augmented visual question answering // Advances
in Neural Information Processing Systems. 36. 2023. 22820–22840.
[33] Lin Zhihong, Zhang Donghao, Tao Qingyi, Shi Danli, Haffari Gholamreza, Wu Qi, He Ming-
guang, Ge Zongyuan . Medical visual question answering: A survey // Artificial Intelligence in
Medicine. 2023. 143. 102611.
[34] Liu Haotian, Li Chunyuan, Li Yuheng, Lee Yong Jae . Improved baselines with visual instruction
tuning // Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
2024. 26296–26306.
[35] Liu Haotian, Li Chunyuan, Wu Qingyang, Lee Yong Jae . Visual instruction tuning // Advances
in neural information processing systems. 36. 2023. 34892–34916.
[36] Liu Ye, Li Hui, Garcia-Duran Alberto, Niepert Mathias, Onoro-Rubio Daniel, Rosenblum
David S . MMKG: multi-modal knowledge graphs // The semantic web: 16th international
conference, ESWC 2019, portorož, Slovenia, June 2–6, 2019, proceedings 16. 2019. 459–474.
[37] Luo Haoran, Chen Guanting, Zheng Yandan, Wu Xiaobao, Guo Yikai, Lin Qika, Feng Yu, Kuang
Zemin, Song Meina, Zhu Yifan, others . HyperGraphRAG: Retrieval-Augmented Generation
with Hypergraph-Structured Knowledge Representation // arXiv preprint arXiv:2503.21322.
2025.
[38] Ma Shengjie, Xu Chengjin, Jiang Xuhui, Li Muzhi, Qu Huaren, Yang Cehao, Mao Jiaxin,
Guo Jian . Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with
Knowledge-guided Retrieval Augmented Generation // The Thirteenth International Conference
on Learning Representations. 2025.
[39] Marino Kenneth, Rastegari Mohammad, Farhadi Ali, Mottaghi Roozbeh . Ok-vqa: A visual
question answering benchmark requiring external knowledge // Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 2019. 3195–3204.
[40] Mensink Thomas, Uijlings Jasper, Castrejon Lluis, Goel Arushi, Cadar Felipe, Zhou Howard,
Sha Fei, Araujo André, Ferrari Vittorio . Encyclopedic vqa: Visual questions about detailed
properties of fine-grained categories // Proceedings of the IEEE/CVF International Conference
on Computer Vision. 2023. 3113–3124.
[41] Ni Bo, Liu Zheyuan, Wang Leyao, Lei Yongjia, Zhao Yuying, Cheng Xueqi, Zeng Qingkai, Dong
Luna, Xia Yinglong, Kenthapadi Krishnaram, others . Towards Trustworthy Retrieval Aug-
mented Generation for Large Language Models: A Survey // arXiv preprint arXiv:2502.06872.
2025.
[42] Ning Liang-bo, Wang Shijie, Fan Wenqi, Li Qing, Xu Xin, Chen Hao, Huang Feiran . Cheatagent:
Attacking llm-empowered recommender systems via llm agent // Proceedings of the 30th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining. 2024. 2284–2295.
[43] Nussbaum Zach, Morris John Xavier, Mulyar Andriy, Duderstadt Brandon . Nomic Embed:
Training a Reproducible Long Context Text Embedder // Transactions on Machine Learning
Research. 2025.
13

[44] Qi Jingyuan, Xu Zhiyang, Shao Rulin, Chen Yang, Di Jin, Cheng Yu, Wang Qifan, Huang
Lifu. RoRA-VLM: Robust Retrieval-Augmented Vision Language Models // arXiv preprint
arXiv:2410.08876. 2024.
[45] Qu Haohao, Ning Liangbo, An Rui, Fan Wenqi, Derr Tyler, Liu Hui, Xu Xin, Li Qing . A survey
of mamba // arXiv preprint arXiv:2408.01129. 2024.
[46] Radford Alec, Kim Jong Wook, Hallacy Chris, Ramesh Aditya, Goh Gabriel, Agarwal Sandhini,
Sastry Girish, Askell Amanda, Mishkin Pamela, Clark Jack, others . Learning transferable visual
models from natural language supervision // International conference on machine learning.
2021. 8748–8763.
[47] Ren Shaoqing, He Kaiming, Girshick Ross, Sun Jian . Faster r-cnn: Towards real-time object
detection with region proposal networks // Advances in neural information processing systems.
28. 2015.
[48] Schwenk Dustin, Khandelwal Apoorv, Clark Christopher, Marino Kenneth, Mottaghi Roozbeh .
A-okvqa: A benchmark for visual question answering using world knowledge // European
conference on computer vision. 2022. 146–162.
[49] Touvron Hugo, Lavril Thibaut, Izacard Gautier, Martinet Xavier, Lachaux Marie-Anne, Lacroix
Timothée, Rozière Baptiste, Goyal Naman, Hambro Eric, Azhar Faisal, others . Llama: Open
and efficient foundation language models // arXiv preprint arXiv:2302.13971. 2023.
[50] Wang P , Bai S, Tan S, Wang S, Fan Z, Bai J, Chen K, Liu X, Wang J, Ge W, others . Qwen2-vl:
Enhancing vision-language model’s perception of the world at any resolution, 2024 // URL
https://arxiv. org/abs/2409.12191. 2024.
[51] Wang Shijie, Fan Wenqi, Feng Yue, Ma Xinyu, Wang Shuaiqiang, Yin Dawei . Knowledge
Graph Retrieval-Augmented Generation for LLM-based Recommendation // arXiv preprint
arXiv:2501.02226. 2025.
[52] Wu Zhiyu, Chen Xiaokang, Pan Zizheng, Liu Xingchao, Liu Wen, Dai Damai, Gao Huazuo,
Ma Yiyang, Wu Chengyue, Wang Bingxuan, others . Deepseek-vl2: Mixture-of-experts vision-
language models for advanced multimodal understanding // arXiv preprint arXiv:2412.10302.
2024.
[53] Yan Yibin, Xie Weidi . EchoSight: Advancing Visual-Language Models with Wiki Knowledge //
Findings of the Association for Computational Linguistics: EMNLP 2024. 2024. 1538–1551.
[54] Yuan Xu, Zhou Li, Sun Zenghui, Zhou Zikun, Lan Jingsong . Instruction-guided multi-granularity
segmentation and captioning with large multimodal model // Proceedings of the AAAI Confer-
ence on Artificial Intelligence. 2025.
[55] Zhang Duzhen, Yu Yahan, Dong Jiahua, Li Chenxing, Su Dan, Chu Chenhui, Yu Dong . MM-
LLMs: Recent Advances in MultiModal Large Language Models // Findings of the Association
for Computational Linguistics: ACL 2024. 2024. 12401–12430.
[56] Zhang Tao, Zhang Ziqi, Ma Zongyang, Chen Yuxin, Qi Zhongang, Yuan Chunfeng, Li Bing,
Pu Junfu, Zhao Yuxuan, Xie Zehua, others . mR2AG: Multimodal Retrieval-Reflection-
Augmented Generation for Knowledge-Based VQA // arXiv preprint arXiv:2411.15041. 2024.
[57] Zhao Zihuai, Fan Wenqi, Li Jiatong, Liu Yunqing, Mei Xiaowei, Wang Yiqi, Wen Zhen, Wang
Fei, Zhao Xiangyu, Tang Jiliang, others . Recommender systems in the era of large language
models (llms) // IEEE Transactions on Knowledge and Data Engineering. 2024.
[58] Zhu Jinguo, Wang Weiyun, Chen Zhe, Liu Zhaoyang, Ye Shenglong, Gu Lixin, Tian Hao, Duan
Yuchen, Su Weijie, Shao Jie, others . Internvl3: Exploring advanced training and test-time
recipes for open-source multimodal models // arXiv preprint arXiv:2504.10479. 2025.
[59] Zhu Xiangrong, Xie Yuexiang, Liu Yi, Li Yaliang, Hu Wei . Knowledge Graph-Guided Retrieval
Augmented Generation // arXiv preprint arXiv:2502.06864. 2025.
14

A. Prompt Design
In our multimodal knowledge graph construction pipeline, we utilize LLMs’ text understanding and
generation capabilities to extract textual knowledge graphs automatically by providing appropriate
prompts. Since previous work [ 15,20] has explored textual KGs extraction, we just follow the prompt
template of LightRAG[20].
The core contribution of our mKG-RAG lies in the challenge of merging textual and visual graphs
into a multimodal graph. To this end, we employ MLLMs as the vision-text matcher, effectively
aligning semantically consistent visual and textual entities/relationships. In this process, we introduce
a well-designed vision-text matching prompt to guide the MLLMs, as shown in Figure 5. Moreover,
several high-quality examples are provided to MLLMs for In-context Learning. One example is
illustrated in Figure 6.
B. Implementation Details
QM-Retriever. In the proposed QM-Retriever, we introduce a Question Converter Fqto transform
interrogative questions into declarative forms, thereby reducing grammatical mismatches with evi-
dence texts. The Question Converter comprises two linear projection layers separated by a ReLU
activation function. This transformation is performed in the latent space, where Fqreformulates the
word embeddings of the original questions into declarative representations before passing them to the
BERT encoder of the Q-Former.
During training, both the Question Converter Fqand the Q-Former are jointly optimized, while the
Visual Encoder Fvis kept frozen. The QM-Retriever is trained for 25 epochs on our annotated dataset
of 221K query-evidence pairs, using the AdamW optimizer and a CosineLR scheduler with an initial
learning rate of 10−5. The training configuration also includes a batch size of 64, a KL divergence
coefficient of 2, an input image size of 224×224, and a maximum token length of 512 for both
questions and evidence.
Fine-turning. Following the experimental setup in ReflectiV A [ 11], we adopt LLaV A-More [ 10] as
our multimodal answer generator. Since ReflectiV A is specifically optimized for relevant passage
filtering and answer generation, we fine-tune our method (mKG-RAG*) accordingly to ensure
consistency with this setup. We employ LoRA adapters [ 24] for parameter-efficient tuning, using
a total batch size of 32 and a learning rate of 1.5×10−4. To preserve the model’s performance on
established MLLM benchmarks, we augment the fine-tuning dataset with samples from the LLaV A-
Instruct-150K dataset [ 35]. Following the strategy of Wiki-LLaV A [ 4], we increase the sampling
probability of these examples, ensuring they comprise approximately half of each mini-batch.
15

Vision-Text Matching Prompt
Based on the provided image, visual scene graph, and textual entities and relationships, match
visual objects/relations in the image with the provided textual entities/relationships.
Input Format :
Each textual entity are formatted as (“entity”| <entity-name> |<entity-type> |
<entity-description> ), which contains the following information:
1)entity-name : Name of the entity;
2)entity-type : Name of the entity type;
3)entity-description : Comprehensive description of the entity’s attributes and activities.
Each textual relationship are formatted as (“relation”| <source-entity> |<target-entity> |
<relation-description> |<relation-strength> ), which contains the following infor-
mation:
1)source-entity : name of the source entity, as defined in the textual entities;
2)target-entity : name of the target entity, as defined in the textual entities;
3)relation-description : explanation as to why the source entity and the target entity
are related to each other;
4)relation-strength : a numeric score indicating the strength of the relationship between
the source and target entities, ranging from 0 to 10.
The scene graph provides the object and relationship information in the image, which is
formatted as:
-<object-0> :<object-category> ,<object-bbox>
-<object-1> :<object-category> ,<object-bbox>
...
-<relation-0> :<object-0> <relation-name> <object-1>
-<relation-2> :<object-1> <relation-name> <object-3>
The<object-bbox> is the bounding box of each object region, represented as (x1, y1, x2,
y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top
left y, bottom right x, and bottom right y.
Matching Steps :
Step 1 . Identify the textual entity that is most relevant to the overall image and extract the
following information:
1)entity-name : the name of the entity that best represents the overall image;
2)strength : a numeric score indicating the strength of the match, ranging from 0 to 10.
Format the image matching as (“matching”| <image> |<entity-name> |<strength> )
Step 2 . For each object in the scene graph, if the object visually depicts a textual entity
identified in the input data, extract the following information:
1)object-id : the ID of the object in the scene graph;
2)entity-name : the name of the entity it represents;
3)strength : a numeric score indicating the strength of the match, ranging from 0 to 10.
Format each object matching as (“matching”| <object-id> |<entity-name> |<strength> )
Step 3 . For each relation in the scene graph, if the relation visually represents a textual
relationship identified in the input data, extract the following information:
1)relation-id : the id of the relation in the scene graph;
2)source-entity : the source entity of the relationship it represents;
3)target-entity : the target entity of the relationship it represents;
4)strength : a numeric score indicating the strength of the match, ranging from 0 to 10.
Format each relation matching as (“matching”| <relation-id> |<source-entity> |
<target-entity> |<strength> )
Step 4 . For those objects or relations without a corresponding text entity or relationship, please
ignore them.
Figure 5: The prompt used to match visual and textual entities/relationships
16

Vision-Text Matching Example
Textual Entities :
(“entity”|MOUNT FUJI|location|Mount Fuji is an active stratovolcano located on Japan’s
Honshu Island, with a peak elevation of 3,776.24 meters. )
(“entity”|HONSHU ISLAND|location|Honshu Island is the largest island of Japan, where
Mount Fuji is situated. )
(“entity”|CHERRY BLOSSOMS|concept|Cherry blossoms are a symbol of Japan, known for
their beauty and cultural significance, often associated with the arrival of spring. )
(“entity”|SHINKANSEN|technology|The Shinkansen, also known as the bullet train, is a
network of high-speed railway lines in Japan. )
Textual Relationships :
(“relationship”|MOUNT FUJI|HONSHU ISLAND|Mount Fuji is located on Honshu Island,
making the island its geographical setting.|9)
(“relationship”|MOUNT FUJI|CHERRY BLOSSOMS|Both Mount Fuji and cherry blossoms
are iconic symbols of Japan, often celebrated together in cultural contexts.|8)
(“relationship”|MOUNT FUJI|SHINKANSEN|Mount Fuji and the Shinkansen are both
recognized as national symbols of Japan.|7)
Image Description : Mount Fuji and the Shinkansen electric car passing in front of it.
Scene Graph :
-<object-0> : train, (0.06, 0.64, 1.0, 0.77)
-<object-1> : fence, (0.0, 0.8, 0.98, 0.88)
-<object-2> : snow, (0.25, 0.29, 0.67, 0.49)
-<object-3> : mountain, (0.0, 0.3, 1.0, 0.64)
-<relation-0> :<object-0> over <object-1>
-<relation-1> :<object-2> on<object-3>
-<relation-2> :<object-3> behind <object-0>
Output :
(“mapping”| <image> |MOUNT FUJI|8)
(“mapping”| <object-3> |MOUNT FUJI|9)
(“mapping”| <object-0> |SHINKANSEN|7)
(“mapping”| <relation-2> |MOUNT FUJI|SHINKANSEN|7)
Figure 6: A high-quality vision-text matching example for In-context Learning
17