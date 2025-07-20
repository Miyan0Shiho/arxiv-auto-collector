# Enhancing Retrieval Augmented Generation with Hierarchical Text Segmentation Chunking

**Authors**: Hai Toan Nguyen, Tien Dat Nguyen, Viet Ha Nguyen

**Published**: 2025-07-14 05:21:58

**PDF URL**: [http://arxiv.org/pdf/2507.09935v1](http://arxiv.org/pdf/2507.09935v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems commonly use chunking strategies
for retrieval, which enhance large language models (LLMs) by enabling them to
access external knowledge, ensuring that the retrieved information is
up-to-date and domain-specific. However, traditional methods often fail to
create chunks that capture sufficient semantic meaning, as they do not account
for the underlying textual structure. This paper proposes a novel framework
that enhances RAG by integrating hierarchical text segmentation and clustering
to generate more meaningful and semantically coherent chunks. During inference,
the framework retrieves information by leveraging both segment-level and
cluster-level vector representations, thereby increasing the likelihood of
retrieving more precise and contextually relevant information. Evaluations on
the NarrativeQA, QuALITY, and QASPER datasets indicate that the proposed method
achieved improved results compared to traditional chunking techniques.

## Full Text


<!-- PDF content starts -->

Enhancing Retrieval Augmented Generation with
Hierarchical Text Segmentation Chunking
Hai-Toan Nguyen⋆, Tien-Dat Nguyen, and Viet-Ha Nguyen
Institute for Artificial Intelligence, VNU University of Engineering and Technology,
Hanoi, 10000, Vietnam
nguyenhaitoan@vnu.edu.vn, datntien@vnu.edu.vn, hanv@vnu.edu.vn
Abstract. Retrieval-Augmented Generation (RAG) systems commonly
use chunking strategies for retrieval, which enhance large language mod-
els (LLMs) by enabling them to access external knowledge, ensuring that
the retrieved information is up-to-date and domain-specific. However,
traditional methods often fail to create chunks that capture sufficient
semantic meaning, as they do not account for the underlying textual
structure. This paper proposes a novel framework that enhances RAG
by integrating hierarchical text segmentation and clustering to generate
moremeaningfulandsemanticallycoherentchunks.Duringinference,the
framework retrieves information by leveraging both segment-level and
cluster-level vector representations, thereby increasing the likelihood of
retrieving more precise and contextually relevant information. Evalua-
tions on the NarrativeQA, QuALITY, and QASPER datasets indicate
that the proposed method achieved improved results compared to tradi-
tional chunking techniques.
Keywords: Retrieval Augmented Generation ·Semantic Chunking ·
Text Segmentation.
1 Introduction
In the field of artificial intelligence (AI), processing unstructured data has be-
come essential. Large Language Models (LLMs), such as OpenAI’s GPT1, is
capable of performing complex tasks and supporting a wide range of applica-
tions [17,18]. While these models are effective at generating responses and au-
tomating tasks, their performance is often limited by the quality of the data they
process.
Updating these models via fine-tuning or other modifications can be chal-
lenging, particularly when dealing with large text corpora [9,16]. One common
approachtoaddressthisissueinvolvesdividinglargevolumesoftextintosmaller,
manageable chunks. This method is widely used in question-answering systems,
where splitting texts into smaller units improves retrieval accuracy [4]. This
retrieval-based method, known as Retrieval-Augmented Generation (RAG) [16],
⋆Corresponding author
1https://openai.com/arXiv:2507.09935v1  [cs.CL]  14 Jul 2025

enhances LLMs by allowing them to reference external knowledge, making it eas-
ier to ensure that the retrieved information is up-to-date and domain-specific.
However, current retrieval-augmented methods have limitations, particularly in
the chunking process. Traditional chunking approaches often fail to capture suf-
ficient semantic meaning as they do not account for the underlying textual
structure. This limitation becomes especially problematic for answering com-
plex queries that require understanding multiple parts of a document, such as
books in NarrativeQA [13] or research papers in QASPER [5].
To address these challenges, we propose a novel framework that improves
the retrieval process by generating chunks that either capture local context (seg-
ments) or clusters of related segments that represent higher-level semantic co-
herence. The key components of this framework are:
1.Text Segmentation : A supervised text segmentation model is applied to
divide the document into smaller, coherent segments. This ensures that each
segmentpreservesmeaningfullocalcontextandisnotcutoffinappropriately.
2.Chunk Clustering : After segmentation, unsupervised clustering combines
related segments based on semantic similarity and their relative positions.
This process creates clusters that maintain sequential structure and capture
broader semantic relationships between segments.
3.Multiple-Vector Based Retrieval : Each text chunk is represented by
multiple vectors: several for individual segments within the chunk, and one
for the cluster itself. This approach provides more options for matching dur-
ing retrieval, as having multiple vectors to compare against increases the
likelihood of finding a more precise match, whether based on specific seg-
ment details or broader cluster context.
2 Related Work
RAG enables LLMs to access external knowledge during inference, improving
their performance on domain-specific tasks. Research by Gao et al. [7] demon-
strates that RAG enhances various Natural Language Processing (NLP) tasks,
particularly in improving factual accuracy. Consequently, several studies have
focused on to enhance RAG’s performance, either by refining the prompts used
for generation [22] or by incorporating diverse retrieval strategies [11].
A key factor in optimizing RAG systems is how documents are chunked
for retrieval. Various traditional chunking methods have been developed, with
open-source frameworks like LangChain2and LlamaIndex3offering techniques
for splitting and filtering documents. Fixed-size chunking [20], which divides
text into equal-length segments, is straightforward to implement but often fails
tocapturethesemanticorstructuralnuancesofthetext.Recursivesplitting[15],
which segments text based on markers like newlines or spaces, can be effective
when documents have clear formatting. Semantic chunking [12], which groups
2https://www.langchain.com/
3https://docs.llamaindex.ai/en/stable/

sentences based on cosine similarity in embedding space, produces more coherent
chunks, but it can sometimes lead to inconsistent boundaries or miss broader
context by focusing too narrowly on sentence-level similarity.
Recent research has made strides in improving chunk retrieval and ensuring
more contextually relevant chunks in RAG systems. LongRAG [10] proposes
using longer retrieval units, such as entire Wikipedia documents, to reduce the
overall size of the corpus. In contrast, RAPTOR [23] creates multi-level chunk
hierarchies, progressing from detailed content at the leaf nodes to more abstract
summaries at higher levels, thereby maintaining relevance while enabling the
model to handle both granular and high-level content.
AnotherareaofimprovementinRAGistheintegrationofKnowledgeGraphs
(KGs) [9]. KGs enhance retrieval accuracy by linking related entities and con-
cepts. For instance, GraphRAG [6] uses entity extraction and query-focused
summarization to organize chunks. However, this can disrupt the natural flow of
the text by grouping chunks from different sections. In contrast, our approach
prioritizes cohesive chunks that maintain the original text structure. By using
unsupervised clustering based on text segmentation, our chunks preserve both
semanticunityandsequentialorder,leadingtoamorecoherentretrievalprocess.
Chunking and text segmentation share similar goals, as both methods aim to
dividetextintomanageable,coherentunits.Earlytextsegmentationapproaches,
such as TextTiling [8] and TopicTiling [21], used unsupervised methods to de-
tect shifts in lexical cohesion. With advancements in neural networks, supervised
modelshaveemerged,suchasSECTOR[1]andS-LSTM[2],whichleverageLong
Short-Term Memory (LSTM) or bidirectional LSTMs to predict segment bound-
aries from labeled data. However, despite these advancements, text segmentation
has traditionally been treated as a standalone task. Our method integrates text
segmentation to improve document chunking, leading to more accurate and co-
herent retrieval.
3 Hierarchical Text Segmentation Framework for RAG
3.1 Overview
Fig. 1 illustrates our proposed RAG framework, which introduces a hierarchical
segmentation and clustering pipeline to enhance chunking accuracy and retrieval
relevance. During the indexing phase, each document Dis segmented into co-
herent segments Si, and related segments are grouped into clusters Cj. Both
segment and cluster embeddings are then computed and stored, as represented
by the following equation:
Indexing (D) ={(Es, Ec)|Es=fsegment (Si), Ec=fcluster (Cj)}(1)
In the retrieval phase, for each chunk Ci, the system computes the cosine
similarity between the query qand all embeddings associated with the chunk,
including both segment embeddings Esand cluster embeddings Ec. The most
relevant embeddings are selected based on similarity scores, calculated as:

Fig. 1.Overview of our framework, incorporating text segmentation during the index-
ing process.
cos(q, Ci) = max( cos(q, Es1), . . . ,cos(q, Esm),cos(q, Ec)) (2)
where Es1, Es2, . . . , E smare segment embeddings for chunk Ci, and Ecis
the cluster embedding for chunk Ci. The system ranks the similarity scores and
selects the top- kchunks for processing by the LLMs to generate the response.
3.2 Text Segmentation and Clustering: A Bottom-Up Approach
In theory, a hierarchical structure would naturally suit a top-down segmenta-
tion approach, where the document is first divided into broader sections and
then broken down into smaller units. However, due to the limitations of cur-
rent text segmentation models—such as the lack of multi-level training data and
difficulties with processing long documents—we propose a bottom-up approach.
This bottom-up approach starts by using supervised methods to identify
smaller, cohesive segments, which are then grouped into larger, meaningful units
Fig. 2.Illustration of the framework’s chunking strategy: The text is first segmented
into coherent parts using a text segmentation model. These segments are then clustered
based on semantic similarities and their sequential order.

through unsupervised clustering techniques. While top-down segmentation may
seem intuitive, especially for capturing hierarchical relationships, current models
struggle with the complexity of longer texts like books or research articles. Fig.
2 shows the structure of this approach. The bottom-up approach works well
with RAG’s retrieval mechanism, which does not depend on a strict sequential
structure. In RAG, chunks are retrieved based on their relevance to the query,
allowing more flexibility in building document representation from smaller units.
Text Segmentation The model used in our research, introduced by Koshorek
et al. [14], is a neural network designed for supervised text segmentation. It
predicts whether a sentence marks the end of a section by using a bidirectional
LSTM to process sentence embeddings. These embeddings are generated in a
previous layer, where another bidirectional LSTM processes the words in each
sentence and applies max-pooling to produce fixed-length representations. The
model is trained to label each sentence as either a ’1’ (indicating the end of
a section) or a ’0’ (indicating continuation), by learning segmentation patterns
from the training data. Fig. 3 demonstrated an example of how a document is
segmented.
Clustering We adapted the clustering method from Glavias et al. [6], where
instead of clustering sentences into segments, we grouped segments into cohesive
clusters. The process is outlined as follows:
1.Graph Construction: After text segmentation, each segment is repre-
sented as a node in a relatedness graph G= (V, E), where Vconsists of
Fig. 3.A biography of Alexander Hamilton is being predicted by a text segmentation
model. The model predicts segment boundaries by labeling each sentence with a 1 or
a 0, where 1 marks the end of a segment and 0 otherwise.

the segments. An edge is added between two segments SiandSjif their
similarity exceeds a predefined threshold. The threshold τis set as:
τ=µ+k·σ (3)
where µis the mean similarity between all segments, σis the standard devi-
ation, and kis a parameter that controls the sensitivity of the connections
between segments.
2.Maximal Clique Detection: The task is to identify all maximal cliques
in the graph G, which are then stored in a set Q.
3.Initial Clustering: An initial set of clusters is created by merging adjacent
segments that are part of at least one clique Q∈Qin graph G
4.Merge Clusters: Adjacent clusters ciandci+1are merged if there is at
least one clique Q∈Qcontaining at least one segment from ciand one from
ci+1. Table 1 provides an illustration of this merging process.
5.Final Merging: Anyremainingsingle-sentenceclustersaremergedwiththe
nearest neighboring cluster, based on cosine similarity, to ensure no isolated
segments remain.
6.Cluster Embedding: Once clusters are finalized, embeddings for each clus-
ter are calculated by applying mean pooling over the vector representations
of the segments within the cluster.
4 Experiments
4.1 Datasets
Our experiments were conducted on three datasets: NarrativeQA, QuALITY,
and QASPER.
NarrativeQA contains 1,572 documents, including books and movie tran-
scripts [13]. The task requires a comprehensive understanding of the entire nar-
rative to answer questions accurately, testing the framework’s ability to compre-
hendandprocesslonger,complextexts.WeassessperformanceusingROUGE-L,
BLEU-1, BLEU-4, and METEOR metrics.
The QuALITY dataset [19] consists of multiple-choice questions, each ac-
companied by a context passage in English. This dataset is particularly useful
for testing the retrieval effectiveness of the system, as answering these ques-
tions often requires reasoning across an entire document. Accuracy is used as
the evaluation metric for this dataset.
Lastly, QASPER is a benchmark dataset designed for question-answering
tasks in scientific NLP papers [5], where the answers are embedded within full-
text documents, rather than being located in a specific section or abstract. Per-
formance is measured using the F1 score.
Table 1. Example of merging segments into clusters from cliques.
Step Sets
Cliques Q{1, 2, 6}, {2, 4, 7}, {3, 4, 5}, {1, 6, 7}
Init. clus. {1, 2}, {3, 4, 5}, {6, 7}
Merge clus. {1, 2, 3, 4, 5}, {6, 7}

4.2 Experimentation Settings
For our experiments, we used the GPT-4o-mini4model as the reader and the
BAAI/bge-m35model for embedding generation. The embeddings were stored
and retrieved using FAISS6, an efficient vector database for similarity search.
We evaluated our chunking strategies by testing different average chunk sizes:
512, 1024, and 2048 tokens. Two retrieval methods were tested: one combining
segment and cluster vector search, and one using cluster vector search alone.
These strategies were compared against fixed-size chunking baselines of 256,
512, 1024, and 2048 tokens, as well as the semantic chunking method by Greg
Kamradt [12], which uses an average embedding size of 256 tokens.
To maintain consistent input size for the LLM across different chunking
strategies, we retrieved a proportional number of chunks based on the segment
length. Specifically, we retrieved 20 chunks for 256-token chunks, 8 chunks for
512-token chunks, 4 chunks for 1024-token chunks, and 2 chunks for 2048-token
chunks. This approach ensured that the total number of tokens retrieved ap-
proximately 4096, allowing for a fair comparison across all chunking strategies.
4.3 Chunking Setup
Text Segmentation As mentioned earlier in section 3.2, the segmentation
model we are using operates on two levels. The sentence embedding level is
a two-layer bidirectional LSTM with an input size of 300 and a hidden size of
256, generating sentence representations through max-pooling over the outputs.
The classifier level is similar to the first, but with double the input size (512),
classifying segment boundaries based on labeled training data.
The model was trained using stochastic gradient descent (SGD) with a batch
size of 32 over 20 epochs, using 100,000 documents from the Wiki727k dataset
[14]. We optimized for cross-entropy loss, classifying whether a sentence marks
the end of a segment. Early stopping was applied after 14 epochs, based on the
validation loss plateauing.
In evaluating the text segmentation model, we used the pkmetric as defined
by Beeferman et al. [3], which measures the probability of error when predict-
ing segment boundaries at random points in the text. Simply put, a lower pk
score means the model is better at accurately identifying where segments end.
The model achieved a pkscore of 35 on the WIKI-50 test set [14], compared
to the original paper’s score of 20. However, it’s important to note that our
focus here is not primarily on optimizing text segmentation but on enhanc-
ing retrieval through segmentation-chunking integration. As a result, we used a
smaller dataset and fewer training epochs, while the original paper used much
larger data and more intensive training. This trade-off allowed us to focus on
the RAG framework while maintaining reasonable segmentation accuracy.
4https://platform.openai.com/docs/models/gpt-4o-mini
5https://huggingface.co/BAAI/bge-m3
6https://github.com/facebookresearch/faiss

Clustering We set the k-values to control the number of clusters, as outlined
in Section 3.2, ensuring alignment with the average chunk size. For an average
of 512 tokens, we used k= 1.2, for 1024 tokens, k= 0.7, and for 2048 tokens,
k= 0.4. Lower k-values directly reduce the number of clusters, resulting in larger
token sizes per cluster.
4.4 Retrieval Results
As shown in Table 2 and Table 3, our segmentation-clustering method performs
better than the other chunking strategies across all three datasets. For Nar-
rativeQA, the average 1024-token segment-cluster method achieves the highest
ROUGE-L score of 26.54, outperforming both the base and semantic methods.
Additionally, it shows an improvement in METEOR score, reaching 30.26. In
the QASPER dataset, the 1024-token segment-cluster method again yields the
best results, with an F1 score of 24.67. Similarly, in the QuALITY dataset,
the segment-cluster method with an average of 512 tokens attains the highest
accuracy of 63.77, outperforming the base 512-token method’s accuracy of 60.23.
While larger chunk sizes, such as the 2048-token configuration, might in-
tuitively seem to provide more context, the results show diminishing returns in
performance. This drop in scores is likely due to the increased size of each chunk,
which makes the chunks more difficult to process and dilutes their coherence. As
chunks grow larger, they capture more information but can become too broad,
causing the reader model to lose focus on query-relevant details.
We believed that using cohesive segments significantly improves the retrieval
process by ensuring that meaningful units of text are retrieved. Traditional
chunking often retrieves fragmented ideas due to arbitrary chunking, resulting in
disjointed answers. Our Segment-Cluster method addresses this issue by group-
ing related segments, even if they are not adjacent. Fig. 4 shown that this ap-
Table 2. Performance on QASPER and QuALITY : Evaluation of various chunk-
ing strategies on the QASPER and QuALITY datasets. The segmentation-clustering
approach yields the highest F1 score on QASPER with 1024-token average segmen-
tation and clustering, and the highest accuracy on QuALITY with 512-token average
segmentation and clustering.
Chunk size Top-k Chunks Methods F1 (QASPER) Accuracy (QuALITY)
256 20Base 19.28 58.16
Semantic 18.07 57.23
512 8Base 20.33 60.23
Cluster Only 21.64 62.36
Segment + Cluster 21.95 63.77
1024 4Base 22.07 58.23
Cluster Only 23.31 58.84
Segment + Cluster 24.67 59.08
2048 2Base 22.05 57.54
Cluster Only 22.76 57.71
Segment + Cluster 23.89 58.85

Table 3. Performance on NarrativeQA : Comparison of various chunking strate-
gies on the NarrativeQA dataset. The 1024-token average segmentation and clustering
outperforms all other chunking strategies across all metrics.
Chunk size Top-k Chunks Methods ROUGE-L BLEU-1 BLEU-4 METEOR
256 20Base 22.21 16.99 5.06 27.11
Semantic 22.5 16.55 5.51 26.56
512 8Base 23.16 17.17 5.77 27.13
Cluster Only 24.12 17.91 6.55 27.56
Segment + Cluster 24.67 18.97 6.83 28.64
1024 4Base 23.86 18.05 6.59 27.12
Cluster Only 25.15 19.28 6.97 29.05
Segment + Cluster 26.54 20.03 7.58 30.26
2048 2Base 23.53 17.65 6.29 27.02
Cluster Only 25.67 19.13 6.80 29.64
Segment + Cluster 26.39 19.62 7.38 30.07
proach captures broader themes, such as training and healthcare factors. While
these factors may not seem relevant to the query at first glance, they together
provide better context. As a result, our method retrieves more coherent and con-
textually relevant information, leading to improved accuracy and overall answer
quality.
The Olympic Gene Pool
Question: The author believes that athletic ability changes over time mainly due
to?
1. Top athletes having fewer children
2. Innate factors
3.Environment
4. Natural selection and genetics
512 Segment + Cluster: It is scarcely surprising that Ethiopian or Kenyan
distance runners do better than everyone else .... Environmental differences be-
tween the two groups could account for differing levels of athletic success ... Better
health care and practicing condition affects athletic ability directly
Answer: 3
Base 512: We know that the inheritance of extra fingers or toes is determined
genetically ... Perhaps way, way back in human history, when our forebears were
still fleeing saber-toothed tigers, natural selection for athletic prowess came into
play... Indeed, the laws of natural selection probably work against athletes these
days.
Answer: 2
Fig. 4.Retrieved chunks based on multiple chunking strategy for the question about
the story The Olympic Gene Pool.

5 Conclusion
This paper introduces a framework that integrates hierarchical text segmenta-
tion with retrieval-augmented generation (RAG) to improve the coherence and
relevance of retrieved information. By combining segmentation and clustering in
chunking, our method ensures that each chunk is semantically and contextually
cohesive, addressing the limitations of traditional, fixed-length chunking.
Experiments showed our approach enhances retrieval accuracy and answer-
ing performance compared to traditional chunking. While a top-down segmen-
tation approach could be ideal, current model limitations favor a bottom-up
combination of supervised and unsupervised techniques. Future work may ex-
plore multi-level segmentation to streamline hierarchical representation in RAG
or use enriched segments in knowledge graph construction to improve entity
relationships and clustering accuracy.
References
1. Arnold, S., Schneider, R., Cudré-Mauroux, P., Gers, F. A., Löser, A.: SECTOR: A
Neural Model for Coherent Topic Segmentation and Classification. Transactions of
the Association for Computational Linguistics, 7:169–184 (2019). doi.org/10.1162/
tacl_a_00261
2. Barrow, J., Jain, R., Morariu, V., Manjunatha, V., Oard, D., Resnik, P.: A Joint
Model for Document Segmentation and Segment Labeling. In: Proceedings of the
58th Annual Meeting of the Association for Computational Linguistics (ACL). pp.
313–322 (2020). doi.org/10.18653/v1/2020.acl-main.29
3. Beeferman, D., Berger, A., Lafferty, J.: Statistical Models for Text Segmentation.
In: Machine Learning 34, 177–210 (1999). doi.org/10.1023/A:1007506220214
4. Chen, D., Fisch, A., Weston, J., Bordes, A.: Reading Wikipedia to Answer Open-
Domain Questions. In: Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pp. 1870–1879 (2017). doi.
org/10.48550/arXiv.1704.00051
5. Dasigi, P., Lo, K., Beltagy, I., Cohan, A., Smith, N. A., Gardner, M.: A Dataset
of Information-Seeking Questions and Answers Anchored in Research Papers. In:
Proceedings of the 2021 Conference of the North American Chapter of the Associa-
tion for Computational Linguistics: Human Language Technologies, pp. 4599–4610
(2021). doi.org/10.18653/v1/2021.naacl-main.365
6. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Larson,
J.: From Local to Global: A Graph RAG Approach to Query-Focused Summariza-
tion (2024). doi.org/10.48550/arXiv.2404.16130
7. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang,
M., Wang, H.: Retrieval-augmented generation for large language models: A survey
(2023). doi.org/10.48550/arXiv.2312.10997
8. Hearst, M. A.: Multi-paragraph segmentation of expository text. In: Proceedings
of the 32nd annual meeting on Association for Computational Linguistics, pp. 9-16
(1994). doi.org/10.3115/981732.981734
9. Ji,S.,Pan,S.,Cambria,E.,Marttinen,P.,Yu,P.S.:ASurveyonKnowledgeGraphs:
Representation, Acquisition, and Applications. In: IEEE Transactions on Neural
Networks and Learning Systems (2021). doi.org/10.1109/TNNLS.2021.3070843

10. Jiang, Z., Ma, X., Chen, W.: LongRAG: Enhancing Retrieval-Augmented Gener-
ation with Long-context LLMs (2024). doi.org/10.48550/arXiv.2406.15319
11. Kalra, R., Wu, Z., Gulley, A., Hilliard, A., Guan, X., Koshiyama, A., Treleaven,
P.: HyPA-RAG: A Hybrid Parameter Adaptive Retrieval-Augmented Generation
System for AI Legal and Policy Applications (2024). doi.org/10.48550/arXiv.2409.
09046
12. Kamradt, G.: 5 Levels Of Text Splitting. https://github.com/FullStackRetrieval-
com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_
Of_Text_Splitting.ipynb. Last accessed 16 August 2024.
13. Kočiský, T., Schwarz, J., Blunsom, P., Dyer, C., Hermann, K. M., Melis, G.,
Grefenstette, E.: The NarrativeQA Reading Comprehension Challenge. In: Trans-
actions of the Association for Computational Linguistics, 6, pp. 317–328 (2018).
doi.org/10.1162/tacl_a_00023
14. Koshorek, O., Cohen, A., Mor, N., Rotman, M., Berant, J.: Text Segmentation as
a Supervised Learning Task. In: Proceedings of the 2018 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Lan-
guage Technologies, Volume 2 (Short Papers), pp. 469–473, Association for Com-
putational Linguistics (2018). doi.org/10.18653/v1/N18-2075
15. LangChain: How to recursively split text by characters. https://python.langchain.
com/docs/how_to/recursive_text_splitter/. Last accessed 16 September 2024.
16. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive NLP tasks. In: Proceedings of the 34th Inter-
national Conference on Neural Information Processing Systems (NIPS ’20). pp.
9459–9474 (2020). doi.org/10.5555/3495724.3496517
17. Minaee, S., Mikolov, T., Nikzad, N., Chenaghlu, M., Socher, R., Amatriain, X.,
Gao, J.: Large Language Models: A Survey (2024). doi.org/10.48550/arXiv.2402.
06196
18. OpenAI: GPT-4 Technical Report (2023). doi.org/10.48550/arXiv.2303.08774
19. Pang, R. Y., Parrish, A., Joshi, N., Nangia, N., Phang, J., Chen, A., Padmaku-
mar, V., Ma, J., Thompson, J., He, H., Bowman, S.: QuALITY: Question Answer-
ing with Long Input Texts, Yes!. In: Proceedings of the 2022 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, pp. 5336–5358, Association for Computational Linguistics
(2022). doi.org/10.18653/v1/2022.naacl-main.391
20. Rahul: A Guide to Chunking Strategies for Retrieval Augmented Generation
(RAG). doi.org/10.5555/12345678
21. Riedl, M., Biemann, C.: TopicTiling: A Text Segmentation Algorithm based on
LDA. In: Proceedings of ACL 2012 Student Research Workshop, pp. 37–42, Asso-
ciation for Computational Linguistics (2012). doi.org/10.5555/23456789
22. Sahoo, P., Singh, A. K., Saha, S., Jain, V., Mondal, S., Chadha, A.: A Systematic
Survey of Prompt Engineering in Large Language Models: Techniques and Ap-
plications. Indian Institute of Technology Patna and Stanford University (2023).
doi.org/10.48550/arXiv.2402.07927
23. Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., Manning, C. D.: RAP-
TOR: Recursive Abstractive Processing for Tree-Organized Retrieval. In: Pro-
ceedings of the International Conference on Learning Representations (2024).
doi.org/10.48550/arXiv.2401.18059