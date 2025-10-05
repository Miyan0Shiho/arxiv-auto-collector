# REBot: From RAG to CatRAG with Semantic Enrichment and Graph Routing

**Authors**: Thanh Ma, Tri-Tam La, Lam-Thu Le Huu, Minh-Nghi Nguyen, Khanh-Van Pham Luu, Huu-Hoa Nguyen

**Published**: 2025-10-02 08:40:55

**PDF URL**: [http://arxiv.org/pdf/2510.01800v1](http://arxiv.org/pdf/2510.01800v1)

## Abstract
Academic regulation advising is essential for helping students interpret and
comply with institutional policies, yet building effective systems requires
domain specific regulatory resources. To address this challenge, we propose
REBot, an LLM enhanced advisory chatbot powered by CatRAG, a hybrid retrieval
reasoning framework that integrates retrieval augmented generation with graph
based reasoning. CatRAG unifies dense retrieval and graph reasoning, supported
by a hierarchical, category labeled knowledge graph enriched with semantic
features for domain alignment. A lightweight intent classifier routes queries
to the appropriate retrieval modules, ensuring both factual accuracy and
contextual depth. We construct a regulation specific dataset and evaluate REBot
on classification and question answering tasks, achieving state of the art
performance with an F1 score of 98.89%. Finally, we implement a web application
that demonstrates the practical value of REBot in real world academic advising
scenarios.

## Full Text


<!-- PDF content starts -->

REBot: From RAG to CatRAG with Semantic
Enrichment and Graph Routing
Thanh Ma, Tam-Tri La, Nghi-Minh Nguyen, Van-Khanh Pham Luu,
Thu-Lam Le Huu, and Huu-Hoa Nguyen∗
{mtthanh,nhhoa}@ctu.edu.vn
{tamb2203579,nghib2203570,vanb2203592,thub2206018}@student.ctu.edu.vn
∗corresponding author
Abstract.Academic regulation advising is vital for helping students
interpret and comply with institutional policies, yet building effective
systems requires domain-specific regulatory resources. To address this
challenge, we propose REBot, an LLM-enhanced advisory chatbot pow-
ered by CatRAG, a hybrid retrieval–reasoning framework that integrates
RAG with GraphRAG. We introduce CatRAG that unifies dense re-
trieval and graph-based reasoning, supported by a hierarchical, category-
labeled knowledge graph enriched with semantic features for domain
alignment. A lightweight intent classifier routes queries to the appro-
priate retrieval modules, ensuring both factual accuracy and contex-
tual depth. We construct a regulation-specific dataset and assess REBot
on classification and question-answering tasks, achieving state-of-the-art
performance with an F1-score of 98.89%. Finally, we implement a web
application that demonstrates the practical value of REBot in real-world
academic advising scenarios.
Keywords:Regulation Advisor·Artificial Intelligence·Chatbot·Text
Classification·Knowledge Graph·RAG·GraphRAG
1 Introduction
Addressing student inquiries regarding university regulations remains a persis-
tent challenge for many Vietnamese universities and institutes. At Can Tho
University (CTU), such information is primarily disseminated through official
sources including the student handbook, departmental websites, and PDF docu-
ments. Consequently, students often face difficulties in locating precise answers,
leading to uncertainty about critical academic procedures and policies. Com-
mon questions include: ’What is the maximum study duration?’, ’What are the
requirements for receiving a university scholarship?’, or ’How can I change my
major?’ To find answers, students may either consult multiple documents or seek
assistance from the relevant department, both of which can be time-consuming
and inefficient. Frequent policy updates further complicate maintaining aware-
ness of the latest regulations.
Artificial intelligence (AI) has been widely adopted in chatbot-based advi-
sory systems. In Vietnam, Phuoc et al. [4] employed Rasa to build a regulationarXiv:2510.01800v1  [cs.AI]  2 Oct 2025

2 Thanh Ma et al.
management chatbot capable of intent detection, entity extraction, and dialogue
management; while effective for FAQs, its heavy reliance on a small dataset lim-
ited robustness for unseen queries. Similarly, Luong and Luong [14] developed
an NLP/ML chatbot for student services, providing fast responses but strug-
gling with policy-specific reasoning. Recent advances in large language models
(LLMs) [1,18,22] such as ChatGPT and Gemini overcome these issues by mod-
eling complex semantics and sustaining multi-turn dialogue, offering valuable
insights for academic advisory systems. Complementary techniques have also
emerged: Retrieval-Augmented Generation (RAG) improves factual grounding
but falters with ambiguous queries, whereas Knowledge Graphs (KGs) capture
structured policy relations but lack generative flexibility. To address these gaps,
we integrate both in GraphRAG, achieving precise, context-aware, and explain-
able responses to complex regulation-related questions.
We present REBot, an LLM-powered chatbot that offers CTU students [15]
fast, accurate, and up-to-date answers on academic regulations. At its core
is CatRAG, a hybrid framework combining RAG for factual reliability with
GraphRAG for structured contextual reasoning. To meet the dual need for di-
rect responses and explanatory context, CatRAG integrates NER for semantic
enrichmentandaroutingclassifierforprecisesubgraphselection,enhancingboth
accuracy and efficiency. Our contributions include: (1) a curated CTU Academic
Regulation Dataset optimized for retrieval; (2) a three-tier Knowledge Graph
linking categories, chunks, and entities for explainable reasoning; (3) an intent
classifier for query routing; (4) CatRAG, a category-guided extension of RAG
with new Construct and Query algorithms; and (5) the REBot web application,
enabling real-time access to regulation guidance.
The primary goal of GraphRAG is not simply retrieving top-ranked answers,
as in conventional RAG, but supplying surrounding contextual knowledge that
enriches responses and yields more comprehensive advice. Yet, applied in isola-
tion,GraphRAGoftenunderperformsstandardRAG[7].Ourapproachtherefore
seeks both to generate accurate answers and to integrate relevant context, en-
hancing the overall quality of academic advising.
2 Background
2.1 Academic Regulations Inquiry
Academic regulations vary across universities, each with distinct administrative
structures and formats. At Can Tho University, regulations are issued under
official decisions and published online1, yet the documents remain fragmented,
unstructured, and written in formal administrative language, making efficient
retrieval difficult for students. To address this, we propose an AI-powered chat-
bot tailored for CTU, leveraging four core techniques: query-oriented text clas-
sification, Named Entity Recognition (NER), Retrieval-Augmented Generation
(RAG), and GraphRAG. These components are elaborated in the following sec-
tions.
1https://dsa.ctu.edu.vn/noi-quy-quy-che/quy-che-hoc-vu.html

REBot: Category-guided GraphRAG for Academic Regulation Advising 3
2.2 Query-Oriented Text Classification for Academic Regulations
Classifying student queries into thematic categories is essential for improving
access to academic regulations. While traditional methods(e.g., Na¨ ıve Bayes,
KNN, SVM)are widely used [17,13], they struggle with the linguistic complex-
ity of natural language. In this work, we take advantage of deep learning models
(e.g., fasttext)for the classification model. Formally, given queriesXand cat-
egoriesY={𝑦 1,...,𝑦𝑘}, a classifierΦ:X→Ymaps each query to its most
relevant category, enabling intent inference and accurate responses. To ensure
coverage, regulatory data are extracted from dispersed PDFs and continuously
updated from CTU’s official sources.
2.3 Semantic Enrichment with NER
Named Entity Recognition (NER) is a central task in information extraction,
identifying entities such as people, organizations, and locations [6,2]. A typi-
cal NER pipeline involves segmentation, tokenization, POS tagging, and entity
detection, transforming unstructured text into structured representations [11].
In this study, we highlight POS tagging as it provides syntactic cues that con-
strain entity search, reduce false positives, and improve accuracy [3,16]. For
Vietnamese, we employUnderthesea2, a robust deep learning toolkit with effi-
cient POS tagging [20]. Formally, given a token sequenceT=(𝑡 1,...,𝑡𝑛), the
NER function mapse=N(T)=(𝑒 1,...,𝑒𝑛), 𝑒𝑖∈E,whereEis the entity
label set. This formulation underscores POS tagging as a vital step that en-
hances NER efficiency and accuracy, thereby improving chatbot responses. The
integration of RAG and GraphRAG as the framework’s core will be detailed in
the following sections.
2.4 Baseline/General Responses using RAG
Large language models (LLMs), while fluent and generalizable, are prone toAI
hallucination3. Retrieval-augmented methods mitigate this by grounding gen-
eration in external knowledge such as corpora, PDFs, or databases. Formally,
given a corpusD={𝐷 1,...,𝐷𝑛}segmented into chunksC={𝑐 1,...,𝑐𝑚}, an
embedding function𝑓 embmaps each chunk intoR𝑑, producing a vector store
V={(𝑐 𝑗, 𝑓emb(𝑐𝑗)) |𝑐𝑗∈ C},which supports efficient similarity-based re-
trieval. We adopt RAG as the primary mechanism, as it anchors responses in
semantically relevant evidence, reducing hallucination and improving factual re-
liability [12]. This retrieval–generation paradigm provides the foundation for
GraphRAG, which further extends capability with structured graph knowledge.
2https://github.com/undertheseanlp/underthesea
3https://cloud.google.com/discover/what-are-ai-hallucinations

4 Thanh Ma et al.
2.5 Enhanced/Extended Responses using GraphRAG
GraphRAG [8] extends RAG by integrating structured knowledge from knowl-
edge graphs (KGs) into retrieval and generation. Unlike standard RAG, which
relies on unstructured corpus, GraphRAG enables entity-centric reasoning and
multi-hop traversal over relational structures [10,12]. As an overview, Figure 1
illustrates the structure of our knowledge graph; its formal definition and role
withinCatRAGwillbedetailedinthealgorithmsection.Thisstructuresupports
fact retrieval, symbolic reasoning, and semantic search [9,23,21], thereby en-
hancing factual consistency and verifiability [24,5]. Hence, we adopt GraphRAG
for extended responses for it enriches search with structured, entity-level knowl-
edge, allowing deeper reasoning beyond unstructured retrieval alone.
Fig.1: Knowledge graph of REBot with 250 nodes
In REBot, this design allows the system to provide domain-specific, context-
aware, and verifiable responses. The subgraph retrieval ensures that answers are
grounded not only in relevant documents but also in the correct entity rela-
tionships, which is essential for regulation-focused dialogue. The next section
explains details about our REBot framework.
3 REBot Framework
We presentREBot, a framework for precise, context-aware responses, initially
built to support CTU students with academic regulations. By combining seman-
tic query analysis with domain-specific context, it offers accurate and adaptable
answers, paving the way for universal next-generation chatbot systems.
The framework comprises five interrelated components: (1)Knowledge Ex-
traction, where PDFs are processed with Docling4for accurate text parsing,
4https://github.com/docling-project/docling

REBot: Category-guided GraphRAG for Academic Regulation Advising 5
Fig.2: The REBot framework
cleaned, segmented, and enriched via NER and relation extraction with Under-
thesea5, then embedded using PhoBERTv2 [19] for reliable Vietnamese represen-
tations; (2)Student Affairs Classification(Φ), in which queries are categorized
into five regulation domains using fastText [25], chosen for its efficiency and ac-
curacy on short Vietnamese texts; (3)REBot Knowledge Graph, a Neo4j vector-
enabled database organizing knowledge into five domain graphs with three lay-
ers (categories, contexts, entities), capturing complex student–affairs relations;
(4)Vectorization and Embeddings, where LlamaIndex6builds a vector index
for RAG, combined with PhoBERTv2 embeddings for semantic similarity, and
GraphRAG integration to unify vector retrieval with graph structures; and (5)
Response Refinement, where Mistral7and ChatGPT8refine retrieved content
into fluent, context-aware answers.
4 CatRAG Algorithm
CatRAG is the core idea of this study. It extends RAG by combining dense
semantic retrieval from a vector store with symbolic reasoning over a regulation-
specific knowledge graph. The prefix “Cat” highlights its category-guided design,
where a classifier routes queries to relevant domains, i.e., enrollment, gradua-
tion, assessment (see Table 1). In CatRAG, documents are chunked, linked to
entities, and organized into subgraphs, unifying unstructured and structured re-
trieval for academic advising. First of all, we formalize CatRAG by introducing
the Regulation-Enriched Graph (REG), a structured representation of academic
regulations. REG maps regulation chunks into entities and triplets, while the
classification modelΦpartitions the graph into category-specific subgraphs. Its
definition is as follows:
5https://github.com/undertheseanlp/underthesea
6https://www.llamaindex.ai/
7https://console.mistral.ai/api-keys
8https://platform.openai.com/docs/api-reference

6 Thanh Ma et al.
Definition 1 (REG).LetC={𝑐 1,...,𝑐𝑀}be a set of document chunks. A
REG is a labeled graph𝐺=(E,R), whereEis the set of entities extracted from
C, andR⊆E×P×Eis the set of relations. Each relation𝑟∈Ris a triplet
(𝑒𝑖,𝑝,𝑒𝑗)with predicate𝑝∈ Plinking entities𝑒 𝑖,𝑒𝑗∈E. Furthermore,𝐺is
partitioned into subgraphs{𝑔 𝑦}, each corresponding to a category label predicted
byΦ.
Before presenting the two CatRAG algorithms, we outline the preprocessing
pipeline that ensures domain-specific accuracy: (i) extracting text from PDFs
with Docling and OCR, (ii) normalizing text via lowercasing, stopword removal,
and LLM-based correction, and (iii) enriching semantics with a custom regula-
tion dictionary containing 46 abbreviation–full term pairs in the academic do-
main. Togerther, these steps enhance entity extraction for the knowledge graph
andsimilaritymatchinginthevectorstore.ThenextsubsectionsdetailCatRAG-
ConstructandCatRAG-Query.
4.1 CatRAG-Construct Algorithm
CatRAG-Constructbuilds a hybrid knowledge base by indexing chunks in a
vector store and linking them to a category-guided knowledge graph. Documents
are chunked, embedded with𝑓 embintoVfor fast semantic retrieval, while each
chunk is classified byΦand attached to its category node in𝐺to ensure domain
locality. Entities extracted viaNare linked to chunks, and relations are added
by an LLM to form typed edges, yielding explainable graph-structured evidence
aligned with the vector index. This design leverages dense retrieval for recall
and graph structure for precision, provenance, and reasoning, with the category
layer narrowing the search space and preventing domain leakage.
Algorithm 1:CatRAG-Construct
Input :D={𝐷 1,...,𝐷𝑛}: Corpus of documents;C={𝑐 1,...,𝑐𝑀}: Corpus of document chunks;
𝑓emb:T→R𝑑: Embedding function mapping text to𝑑-dimensional vectors;𝐺: Knowledge
graph;V: Vector store;N: Named entity recognition-POS tagging model;Φ: Text classifier
model;LLM: Large language model for relation extraction
Output:𝐺,V
1 begin
2V←∅,𝐺←∅
3 for𝐷∈Ddo
4segment𝐷into chunks{𝑐 1,...,𝑐𝑘}and add toC
5 for𝑐∈Cdo
6v 𝑐←𝑓emb(𝑐text)
7store(𝑐id,𝑐text,v𝑐)inV
8𝑙𝑎𝑏𝑒𝑙 𝑐←Φ(𝑐 text)
9attach𝑐to category𝑙𝑎𝑏𝑒𝑙 𝑐in𝐺
10𝑇←tokenize(𝑐 text)
11E 𝑐←N(𝑇)
12 for𝑒∈E 𝑐do
13add node𝑒to𝐺; create edge(𝑐)-[:MENTIONS]−(𝑒)
14R 𝑐←LLM.extract_relations(𝑐 text,E𝑐)
15 for(𝑒 𝑖,𝑝,𝑒𝑗)∈R𝑐do
16add edge(𝑒 𝑖)-[:RELATED_TO {predicate}=p, chunk_id=𝑐id]−(𝑒𝑗)to𝐺
17 return𝐺,V
In terms of complexity, preprocessing and chunking are linear in corpus size;
embedding𝑀chunks costs𝑂(𝑀𝑑); classification adds𝑂(𝑀𝐶 Φ); NER adds

REBot: Category-guided GraphRAG for Academic Regulation Advising 7
𝑂(Í
𝑐|𝑐|); and LLM-based relation extraction contributesÍ
𝑐𝐶RE(|𝑐|), typically
dominant. Space usage is𝑂(𝑀𝑑)for the vector index and𝑂(|𝑉|+|𝐸|)for the
graph. Practical considerations include choosing chunk size to balance recall vs.
redundancy, rate-limiting LLM relation extraction, and enforcing schema/typing
to control graph sparsity and noise.
4.2 CatRAG-Query Algorithm
The core idea ofCatRAG-Query(Algorithm 2) is to unify vector-based retrieval
with category-guided graph retrieval, balancing recall efficiency with domain-
specific precision. A user query is embedded and classified into an academic
category, then matched against top-𝑘 vecresults from the global vector store and
top-𝑘 graphchunks from the corresponding knowledge subgraph. Entities and re-
lations are expanded to enrich evidence, which is passed with the query to an
LLM for final answer generation. This hybrid design addresses the limits of vec-
tor retrieval (low explainability) and pure graph retrieval (over-restrictiveness),
yielding context-aware, accurate, and explainable responses.
Algorithm 2:CatRAG-Query
Input :𝑞: user query,𝑓emb:T→R𝑑: embedding function mapping text to𝑑-dimensional vectors,V:
vector store,𝐺: knowledge graph, partitioned into labeled subgraphs{𝑔 𝑦},Φ: classification
model mapping text to labels,LLM: large language model for context synthesis,𝑘 vec: number of
top vector matches.
Output:𝑟𝑒𝑝: generated answer
1 begin
2v𝑞←−𝑓emb(𝑞)
3𝑙𝑎𝑏𝑒𝑙 𝑞←−Φ(𝑞)
4C vec←−TopK(V,v 𝑞,𝑘vec)
5Cgraph←−∅,Egraph←−∅,Rgraph←−∅
6 for𝑔 𝑦∈𝐺do
7 if𝑦=𝑙𝑎𝑏𝑒𝑙 𝑞then
8Cgraph←−TopKChunks(𝑔 𝑦,v𝑞,𝑘graph)
9 for𝑐∈Cgraphdo
10Egraph←−Egraph∪Entities(𝑐)
11Rgraph←−Rgraph∪Relations(Egraph,𝑔𝑦)
12Cfinal←−Cvec∪Cgraph
13𝑐𝑜𝑛𝑡𝑒𝑥𝑡←−concat(Cfinal,Egraph,Rgraph)
14𝑟𝑒𝑝←−LLM.generate(𝑞,𝑐𝑜𝑛𝑡𝑒𝑥𝑡)
15 return𝑟𝑒𝑝
In terms of complexity, query embedding and classification are linear in the
query length, vector and graph retrieval require𝑂(𝑁 𝑣𝑑+𝑛𝑦𝑑)with brute-force
search (or𝑂(log𝑁 𝑣+log𝑛𝑦)with approximate nearest neighbor search), and
entityexpansioncosts𝑂(𝑘 graph(¯𝑒+¯𝑟)),where ¯𝑒and ¯𝑟denotetheaveragenumbers
of entities and relations per chunk. The LLM generation cost𝐶 gen(𝐿)usually
dominates in practice. A key consideration is that large vector stores or dense
subgraphs can significantly increase retrieval overhead, while excessive entity
expansion may inflate the context length and thus raise the LLM cost. Hence,
careful tuning of𝑘 vec,𝑘graph, and context size is crucial to balance efficiency and
answer quality.

8 Thanh Ma et al.
5 Dataset and Experimental Result
5.1 Dataset and Implementation Environment
We compiled two distinct datasets to support our study. The first dataset com-
prises1,319question–answer (Q&A) pairs, carefully curated to evaluate the
chatbot’s ability to address inquiries related to academic regulations. The sec-
ond dataset contains3,256questions developed for training a FastText-based
classification model. This classification dataset was divided into training and
testing subsets using an80:20split. Most of the data were sourced from the offi-
cial departmental websites of Can Tho University9and other reputable sources.
To ensure reliability, all datasets underwent rigorous manual verification. Fur-
thermore, all responses were authored by academic support experts at Can Tho
University. The training data and source code of our implementation are pub-
licly available on GitHub10. The experiments were conducted on a computer
equipped with an Intel(R) Core (TM) i5-12500H 4.5 GHz processor, 16 GB of
RAM, 6 GB of vRAM and running Windows 11 OS. Next, we will present the
experimental results of our approach in the subsequent subsection.
Table 1: Collected Dataset for REBot.
IDName Number of Note
Samples
Specialized Domains Classification
1Study and Training 474 Học tập và rèn luyện
2Education 611 Đào tạo
3Domitory 655 Ký túc xá
4Discipline and Scholarships 477 Khen thưởng và kỷ luật
5Graduation 485 Tốt nghiệp
6Other 550 Khác
Total: 3,252
REBot Q&A Evaluation
1Truth 909 Knowledgerelatedto CTU academic regulation
2Other 404 Knowledgeoutsideof CTU academic regulation
Total: 1,313
5.2 Quantitative Results
WeinstructREBotmotivatedfromtwoarchitecturalvariants:thestandardRAG
and the graph-enhanced CatRAG (see Figure 2). As previously outlined, (1) the
system utilizes phobert-base-v2 as the embedding model and retrieves the top
five most relevant chunks to provide contextual input for the generative model.
(2) For the CatRAG variant, a similarity threshold of 0.7 is applied when lever-
aging the knowledge graph to ensure semantically relevant node connections.
9https://www.ctu.edu.vn/don-vi-truc-thuoc.html
10https://github.com/tamB2203579/GraphRAG

REBot: Category-guided GraphRAG for Academic Regulation Advising 9
Table 2: Comparison of CatRAG (CR) and RAG (R) across thresholds
Provider Threshold Accuracy Precision Recall F1 Score Eval. Time
(CR | R) (CR | R) (CR | R) (CR | R) (CR | R)
OpenAI0.698.48 | 98.17 99.33 | 98.78 98.46 | 98.56 98.89 | 98.67 – | –
0.795.58 | 93.75 95.11 | 92.32 98.39 | 98.46 96.72 | 95.29 2h10m12s | 2h10m12s
0.888.58 | 86.44 84.87 | 81.65 98.20 | 98.26 91.05 | 89.19 – | –
Mistral AI0.698.25 | 98.55 99.11 | 99.33 98.35 | 98.57 98.73 | 98.95 – | –
0.795.66 | 93.99 95.33 | 92.67 98.28 | 98.47 96.79 | 95.48 1h52m47s | 1h47m17s
0.885.00 | 82.86 79.76 | 76.42 97.95 | 98.14 87.92 | 85.93 – | –
Table 3: Computation Time of CatRAG and RAG (average of 10 runs).
ID Model Name CatRAG (s) RAG (s)
1gpt-4o-mini (OpenAI) 7.2577 5.1076
2mistral-small-2506 (Mistral) 7.0232 4.7692
Table 4: Results of Specialized Domain/Topic Classification.
ID Models Parameters AccPrecRecF1Time (seconds)
(%)(%)(%)(%)Training Testing
1Multinomial NB 𝑎𝑙𝑝ℎ𝑎=0.1 87.1287.1687.1286.780.03 0.01
2 KNN 𝑘=101 83.7485.6183.7482.070.030.183
3Logistic Regression 𝐶=10,𝑡𝑜𝑙=0.0001 93.8793.9593.8793.811.160.009
4 SVM 𝐶=1,𝑡𝑜𝑙=0.0001,𝑙𝑜𝑠𝑠=ℎ𝑖𝑛𝑔𝑒 91.4191.2391.4191.020.160.010
5Random Forest 𝑛_𝑒𝑠𝑡𝑖𝑚𝑎𝑡𝑜𝑟𝑠=300 94.7994.9294.7994.660.790.186
6FastText 𝑒𝑝𝑜𝑐ℎ𝑒𝑠=100,𝑁𝑔𝑟𝑎𝑚𝑠=3 95.7595.7695.7595.672.2775 0.056
The experiments are conducted using two generative AI models:gpt-4o-mini
and mistral-small-2506. To assess the performance of both architectures under
different threshold configurations, we employ four core evaluation metrics: Ac-
curacy, Precision, Recall, and F1 Score. Among these, the F1 Score is prioritized
as the primary criterion for model selection due to its balanced integration of
Precision and Recall.
The results of both architectures are presented in Table 2. Among the config-
urations, CatRAG combined with gpt-4o-mini demonstrates the most impressive
performance, consistently outperforming its RAG counterpart across all thresh-
old settings. Notably, it achieves a near 98.9% F1 Score at a threshold of 0.6,
while still maintaining over 90% F1 Score at a threshold of 0.8, indicating strong
robustnessacrossdifferentsensitivitylevels.Incontrast,MistralAIexhibitsmore
variability across thresholds. While the performance at thresholds 0.6 and 0.7
is comparatively lower, it achieves a notable improvement at threshold 0.8, sug-
gesting that the model performs better when operating under stricter filtering
conditions. Note that, we conduct a gridsearch from0.1to1.0. Then,[0.6,0.8]
obtained the high accuracy.
Table 3 shows that CatRAG consistently incurs longer response times than
standard RAG due to graph-based retrieval and node similarity computation.
For instance, with gpt-4o-mini, CatRAG averages 7.26 s per query versus 5.11

10 Thanh Ma et al.
s for RAG, and with mistral-small-2506, 7.02 s versus 4.77 s. Despite this∼2 s
overhead,latencyremainsacceptableforreal-timeusegivenCatRAG’senhanced
retrieval capability. Table 4 reports classification results, where Random Forest
(94.79%accuracy,94.66%F1)andLogisticRegression(93.87%,93.81%)perform
strongly among traditional models, but KNN lags (83.74%, 82.07%). FastText
achieves the best performance (95.75% accuracy, 95.67% F1), demonstrating
superior ability to capture semantic features while maintaining efficiency.
5.3 Qualitative Result
The questions and answers are expressed with clarity and effectively address
users’ needs. Illustrative examples are presented as follows:
–Question 1:Input:How many semesters are there in an academic year?
Output:An academic year at Can Tho University consists of three semesters, called semester 1, semester 2, and semester 3.
Each semester lasts 15 weeks, including time for teaching, examinations, and result processing.
–Question 2:
Input:What are the foreign language graduation requirements for students?
Output:The foreign language graduation requirements at Can Tho University include:
1.Meeting language proficiency standards: Students must reach the required foreign language level according to their training
program. Specifically:
•Students of cohort 45 and later must have a foreign language certificate at level 3/6 according to the Vietnamese
6-level language proficiency framework or equivalent.
•Students of cohort 44 and earlier must have a French certificate level 2 or above, or an equivalent English certificate.
2.Participation in intensive language programs: Students need to join intensive language programs organized by the university
to improve their foreign language skills to the minimum required level.
3.Exemption from language courses: Some students may be exempt from basic language courses if they hold an appropriate
certificate or passed the language proficiency test at the beginning of the course, as decided by the language proficiency
council.
In summary, to graduate, students must complete the foreign language requirements of their training program.
In this paper, we distilled four key questions reflecting practical concerns
of Can Tho University students. The responses were largely accurate and con-
textually enriched, though certain details require further refinement. Overall,
the information is sufficient for users to draw informed conclusions. Finally, we
present the REBot interface and discuss its broader implications.
5.4 Application and Discussion
After training the AI models, we deploy the framework on a web-based platform
integratedwiththeREBotinterface(Figure3).Thesystemisimplementedusing
modern libraries and frameworks, including React (19.0.0), ViteJS (6.3.1), and
Bootstrap (5.3).
REBot demonstrates effective real-time conversational interaction with stu-
dents; however, several limitations remain. (1) GraphRAG-based retrieval can
introduce excessive context, leading to incoherent or inaccurate responses. (2)
dataset classification requires improvement for handling long-context queries.
(3) the current average response time of about seven seconds is a significant
bottleneck. As future work, we plan to optimize retrieval strategies and reduce
latency, while expanding the dataset vertically to capture finer-grained regula-
tory details within each academic domain (e.g., specific course requirements, ex-
ceptions, procedural workflows). Finally, we will be refining the domain-specific
dictionary, enhancing classification models, and incorporating adaptive retrieval
with reranking for contextual precision.

REBot: Category-guided GraphRAG for Academic Regulation Advising 11
Fig.3: The Interface of our REBot System
6 Conclusion
This study presents the effectiveness of a retrieval-augmented chatbot system
using standard Retrieval-Augmented Generation (RAG) and knowledge-graph-
enhancedGraphRAGtoprovideaccurateanddetailedresponsestostudentques-
tionsatCanThoUniversity.QuantitativetestsindicatethatCatRAG,combined
with a compact generative model, achieves top performance with an F1-score
98.89%, while keeping computation times practical for real-time use. Qualita-
tive reviews confirm the system’s value in offering clear, in-depth explanations
that go beyond simple queries to improve user understanding. By incorporat-
ing advanced embedding techniques and knowledge graphs, this approach beats
traditional methods and lays the groundwork for flexible, domain-focused AI
helpers in education.
Acknowledgements:This research is part of Can Tho University’s scientific
research program under the code THS2025-69.
References
1. A survey on large language model (llm) security and privacy: The good, the bad,
and the ugly. High-Confidence Computing4(2), 100211 (2024)
2. Al-Moslmi, T., Gallofré Oca˜ na, M., L. Opdahl, A., Veres, C.: Named entity ex-
traction for knowledge graphs: A literature overview. IEEE Access8, 32862–32881
(2020)
3. Chiche, A., Yitagesu, B.: Part of speech tagging: a systematic review of deep learn-
ing and machine learning approaches. Journal of Big Data9(1), 10 (2022)
4. Doan, H., Le, V., van, K.: Xây dựng khung ứng dụng ai chatbot trong lĩnh vực
quy chế đào tạo. Hue University Journal of Science: Techniques and Technology
131, 39–52 (06 2023)

12 Thanh Ma et al.
5. Fatemi, B., Halcrow, J., Perozzi, B.: Talk like a graph: Encoding graphs for large
language models. arXiv preprint arXiv:2310.04560 (2023)
6. Goyal, A., Gupta, V., Kumar, M.: Recent named entity recognition and classifica-
tion techniques: a systematic review. Computer Science Review29, 21–43 (2018)
7. Han, H., Shomer, H., Wang, Y., Lei, Y., Guo, K., Hua, Z., Long, B., Liu, H., Tang,
J.: Rag vs. graphrag: A systematic evaluation and key insights. arXiv preprint
arXiv:2502.11371 (2025)
8. Han, H., Wang, Y., Shomer, H., Guo, K., Ding, J., Lei, Y., Halappanavar, M.,
Rossi, R.A., Mukherjee, S., Tang, X., et al.: Retrieval-augmented generation with
graphs (graphrag). arXiv preprint arXiv:2501.00309 (2024)
9. Hogan, A., Blomqvist, E., Cochez, M., et al, D.: Knowledge graphs. ACM Com-
puting Surveys54(4), 1–37 (Jul 2021)
10. Hu, Y., Lei, Z., Zhang, Z., Pan, B., Ling, C., Zhao, L.: Grag: Graph retrieval-
augmented generation. arXiv preprint arXiv:2405.16506 (2024)
11. Huyen, N.T.M., Luong, V.X.: Vlsp 2016 shared task: Named entity recognition.
Proceedings of Vietnamese Speech and Language Processing (VLSP) (2016)
12. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨ uttler, H.,
Lewis, M., tau Yih, W., Rockt¨ aschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive nlp tasks (2021)
13. Li,Q.,Peng,H.,Li,J.,Xia,C.,Yang,R.,Sun,L.,Yu,P.S.,He,L.:Asurveyontext
classification: From traditional to deep learning. ACM Transactions on Intelligent
Systems and Technology (TIST)13(2), 1–41 (2022)
14. Luong, H., Luong, K.: A chatbot-based academic advising model for student in
informationtechnology:Acasestudy.SaudiJournalofEngineeringandTechnology
10(3), 93–100 (2025)
15. Ma, T., Chau, T.K., Thai, P.A., Tram, T.M., Huynh, K., Tran-Nguyen, M.T.:
Racos: Ai-routed chat-voice admission consulting support system. In: International
Conference on Intelligent Systems and Data Science. pp. 295–310. Springer (2024)
16. Minh, P.Q.N.: A feature-rich vietnamese named-entity recognition model (2018),
https://arxiv.org/abs/1803.04375
17. Miro´ nczuk, M.M., Protasiewicz, J.: A recent overview of the state-of-the-art ele-
ments of text classification. Expert Systems with Applications106, 36–54 (2018)
18. Nazir, A., Wang, Z.: A comprehensive survey of chatgpt: Advancements, applica-
tions, prospects, and challenges. Meta-Radiology1(2), 100022 (2023)
19. Nguyen, D.Q., Nguyen, A.T.: Phobert: Pre-trained language models for viet-
namese. arXiv preprint arXiv:2003.00744 (2020)
20. Nguyen, D.Q., Vu, T., Nguyen, D.Q., Dras, M., Johnson, M.: From word segmen-
tation to pos tagging for vietnamese. arXiv preprint arXiv:1711.04951 (2017)
21. Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., Wu, X.: Unifying large language
models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and
Data Engineering36(7), 3580–3599 (2024)
22. Pande, A., Patil, R., Mukkemwar, R., Panchal, R., Bhoite, S.: Comprehensive
study of google gemini and text generating models: Understanding capabilities
and performance (11 2024)
23. Peng, C., Xia, F., Naseriparsa, M., Osborne, F.: Knowledge graphs: Opportunities
and challenges. Artificial Intelligence Review56(11), 13071–13102 (2023)
24. Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y., Ni, L.M., Shum, H.Y., Guo,
J.: Think-on-graph: Deep and responsible reasoning of large language model on
knowledge graph. arXiv preprint arXiv:2307.07697 (2023)
25. Yao, T., Zhai, Z., Gao, B.: Text classification model based on fasttext. ICAIIS’20
pp. 154–157 (2020)